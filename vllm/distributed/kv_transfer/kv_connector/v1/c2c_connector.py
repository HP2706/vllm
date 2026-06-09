# SPDX-License-Identifier: Apache-2.0
"""C2C (Cache-to-Cache) KV transfer connector prototype.

Phase A: raw cross-server KV transfer over CUDA IPC with timing.
Phase B: consumer-side semantic fusion using the trained C2C projectors
from thu-nics/C2C (nics-efc/C2C_Fuser checkpoints).

Producer (sharer model server): after prefill, extracts the prompt's KV from
the paged cache for every attention layer, stacks it into one staging tensor,
and publishes a CUDA IPC handle keyed by a hash of the prompt token ids.

Consumer (receiver model server): pulls the sharer KV device-to-device inside
its own prefill forward, then at the end of the forward (when its own prompt
KV exists in the paged cache) applies the per-layer C2C projector and writes
the fused KV back into its paged blocks. Fusion is pointwise per position and
causal, so it composes with prefix caching: a follow-up request with the same
prompt reuses the fused blocks, making every decode token attend to fused KV
(the offline RosettaModel additionally recomputes the prompt hidden states
over fused KV; that recompute pass is intentionally skipped here).

Consumer extra config:
  c2c_dir: handshake/staging dir (default /dev/shm/c2c_kv)
  c2c_checkpoints_dir: path to the C2C fuser "final" dir with
    projector_{i}.json/.pt and projector_config.json. If unset, the consumer
    only transfers (Phase A behavior) and never mutates its cache.
"""

import json
import os
import pickle
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch.multiprocessing.reductions import reduce_tensor

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _prompt_key(token_ids: list[int]) -> str:
    import hashlib

    h = hashlib.sha256(b",".join(str(t).encode() for t in token_ids))
    return h.hexdigest()[:24]


@dataclass
class C2CReqMeta:
    req_id: str
    key: str
    num_prompt_tokens: int
    num_computed_tokens: int
    slot_mapping: torch.Tensor  # full prompt slots, length num_prompt_tokens
    is_store: bool


@dataclass
class C2CConnectorMetadata(KVConnectorMetadata):
    requests: list[C2CReqMeta] = field(default_factory=list)


class C2CConnector(KVConnectorBase_V1):
    """Prototype connector for C2C-style cross-model KV transfer + fusion."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size: int = vllm_config.cache_config.block_size
        self._is_producer: bool = self._kv_transfer_config.kv_role == "kv_producer"
        self._dir: str = self._kv_transfer_config.get_from_extra_config(
            "c2c_dir", "/dev/shm/c2c_kv"
        )
        self._checkpoints_dir: str | None = self._kv_transfer_config.get_from_extra_config(
            "c2c_checkpoints_dir", None
        )
        os.makedirs(self._dir, exist_ok=True)
        # producer worker state: key -> staging tensor (kept alive for IPC)
        self._staged: dict[str, torch.Tensor] = {}
        # per-step accumulation: key -> list[(layer_name, kv_tensor)]
        self._pending_layers: dict[str, list[tuple[str, torch.Tensor]]] = {}
        # consumer worker state
        self._transferred: set[str] = set()
        self._fused_ranges: dict[str, int] = {}  # key -> tokens fused so far
        self._fuse_jobs: list[C2CReqMeta] = []
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._projectors: list[torch.nn.Module] | None = None
        self._layer_mapping: dict[int, tuple[int, int]] = {}
        self._timings: dict[str, dict[str, Any]] = {}
        logger.info(
            "C2CConnector role=%s kv_role=%s dir=%s fuse=%s",
            role,
            self._kv_transfer_config.kv_role,
            self._dir,
            bool(self._checkpoints_dir),
        )

    # ------------------------------------------------------------------
    # Scheduler-side
    # ------------------------------------------------------------------

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        # Both roles compute their own prefill; we never claim external tokens.
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        return

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = C2CConnectorMetadata()
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = list(new_req.prompt_token_ids or [])
            if not token_ids:
                continue
            num_scheduled = scheduler_output.num_scheduled_tokens[new_req.req_id]
            num_computed = new_req.num_computed_tokens
            # Only handle prefills completing in this single step (allowing a
            # prefix-cached prefix). Chunked prefill is out of scope.
            if num_computed + num_scheduled < len(token_ids):
                logger.warning(
                    "C2C: skipping chunked prefill request %s (%d+%d < %d)",
                    new_req.req_id,
                    num_computed,
                    num_scheduled,
                    len(token_ids),
                )
                continue
            block_ids = list(new_req.block_ids[0])
            block_ids_t = torch.tensor(block_ids, dtype=torch.long)
            offsets = torch.arange(0, self._block_size, dtype=torch.long)
            slot_mapping = (
                offsets.reshape(1, -1) + block_ids_t.reshape(-1, 1) * self._block_size
            ).flatten()[: len(token_ids)]
            meta.requests.append(
                C2CReqMeta(
                    req_id=new_req.req_id,
                    key=_prompt_key(token_ids),
                    num_prompt_tokens=len(token_ids),
                    num_computed_tokens=num_computed,
                    slot_mapping=slot_mapping,
                    is_store=self._is_producer,
                )
            )
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None

    # ------------------------------------------------------------------
    # Worker-side
    # ------------------------------------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._kv_caches = kv_caches

    def _ordered_layer_names(self) -> list[str]:
        def layer_idx(name: str) -> int:
            m = re.search(r"layers\.(\d+)\.", name)
            assert m is not None, f"C2C: cannot parse layer index from {name}"
            return int(m.group(1))

        return sorted(self._kv_caches.keys(), key=layer_idx)

    def _ensure_projectors(self, device: torch.device) -> None:
        if self._projectors is not None:
            return
        assert self._checkpoints_dir is not None
        from rosetta.model.projector import load_projector

        cfg_path = os.path.join(self._checkpoints_dir, "projector_config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        # {"0": {"1": {target_layer: [[source_layer, projector_idx]]}}}
        inner = cfg["0"]["1"]
        self._layer_mapping = {
            int(t): (int(pairs[0][0]), int(pairs[0][1])) for t, pairs in inner.items()
        }
        num = len(
            [
                f
                for f in os.listdir(self._checkpoints_dir)
                if re.match(r"projector_\d+\.pt", f)
            ]
        )
        projectors: list[torch.nn.Module] = []
        for i in range(num):
            proj = load_projector(
                os.path.join(self._checkpoints_dir, f"projector_{i}.json")
            )
            state = torch.load(
                os.path.join(self._checkpoints_dir, f"projector_{i}.pt"),
                map_location=device,
            )
            proj.load_state_dict(state, strict=False)
            proj = proj.to(device=device, dtype=torch.bfloat16).eval()
            projectors.append(proj)
        self._projectors = projectors
        logger.info(
            "C2C consumer: loaded %d projectors, mapping for %d receiver layers",
            num,
            len(self._layer_mapping),
        )

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        if self._is_producer:
            return
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, C2CConnectorMetadata)
        for req in metadata.requests:
            if req.is_store:
                continue
            if req.key not in self._transferred:
                manifest_path = os.path.join(self._dir, f"{req.key}.manifest")
                if not os.path.exists(manifest_path):
                    continue
                t0 = time.perf_counter()
                with open(manifest_path, "rb") as f:
                    manifest = pickle.load(f)
                rebuild_fn, rebuild_args = manifest["ipc"]
                remote: torch.Tensor = rebuild_fn(*rebuild_args)
                t1 = time.perf_counter()

                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                local = torch.empty_like(remote)
                local.copy_(remote)
                end_evt.record()
                end_evt.synchronize()
                copy_ms = start_evt.elapsed_time(end_evt)

                nbytes = local.numel() * local.element_size()
                self._transferred.add(req.key)
                self._staged[req.key] = local
                self._timings[req.key] = {
                    "key": req.key,
                    "num_prompt_tokens_consumer": req.num_prompt_tokens,
                    "producer_shape": list(local.shape),
                    "bytes": nbytes,
                    "manifest_open_ms": (t1 - t0) * 1000.0,
                    "ipc_copy_ms": copy_ms,
                    "bandwidth_gb_s": (nbytes / 1e9) / (copy_ms / 1000.0)
                    if copy_ms > 0
                    else None,
                    "producer_extract_ms": manifest["extract_ms"],
                    "fusion_ms": None,
                }
                if self._checkpoints_dir is None:
                    out_path = os.path.join(self._dir, f"timing_{req.key}.json")
                    with open(out_path, "w") as f:
                        json.dump(self._timings[req.key], f, indent=2)
                logger.info(
                    "C2C consumer: transferred %.2f MB for key=%s in %.3f ms",
                    nbytes / 1e6,
                    req.key,
                    copy_ms,
                )
            # Queue fusion for this request (runs in wait_for_save, once the
            # receiver's own prompt KV for this step exists in the paged cache).
            if self._checkpoints_dir is not None and req.key in self._staged:
                self._fuse_jobs.append(req)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        if not self._is_producer:
            return
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, C2CConnectorMetadata)
        for req in metadata.requests:
            if not req.is_store or req.key in self._staged:
                continue
            # Standard (non-MLA) layout: [2, num_blocks, block_size, H, D]
            assert kv_layer.dim() >= 4 and kv_layer.shape[0] == 2, (
                f"C2C: unexpected KV layout {tuple(kv_layer.shape)} for "
                f"{layer_name}; only standard FlashAttention layout supported"
            )
            num_blocks, block_size = kv_layer.shape[1], kv_layer.shape[2]
            slot_mapping = req.slot_mapping.to(kv_layer.device)
            flat = kv_layer.reshape(2, num_blocks * block_size, *kv_layer.shape[3:])
            extracted = flat[:, slot_mapping].clone()
            self._pending_layers.setdefault(req.key, []).append(
                (layer_name, extracted)
            )

    @torch.inference_mode()
    def _run_fusion(self, req: C2CReqMeta) -> None:
        staged = self._staged[req.key]  # [L_src, 2, T, H_s, D_s]
        device = staged.device
        self._ensure_projectors(device)
        assert self._projectors is not None

        t_total = staged.shape[2]
        if t_total != req.num_prompt_tokens:
            logger.warning(
                "C2C: token count mismatch producer=%d consumer=%d key=%s; "
                "skipping fusion",
                t_total,
                req.num_prompt_tokens,
                req.key,
            )
            return
        # Fuse positions [start, T-1): resume after any prefix-cached (already
        # fused) prefix, and exclude the final prompt position, matching the
        # offline kv_cache_index convention ([1,0]... [-1,0]).
        already = self._fused_ranges.get(req.key, 0)
        start = max(req.num_computed_tokens, already)
        end = req.num_prompt_tokens - 1
        if start >= end:
            return
        positions = torch.arange(start, end, device=device)
        slots = req.slot_mapping.to(device)[start:end]

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        layer_names = self._ordered_layer_names()
        assert len(layer_names) == len(self._layer_mapping), (
            f"C2C: receiver has {len(layer_names)} attention layers but "
            f"mapping covers {len(self._layer_mapping)}"
        )
        for tgt_layer, layer_name in enumerate(layer_names):
            src_layer, proj_idx = self._layer_mapping[tgt_layer]
            proj = self._projectors[proj_idx]
            kv_layer = self._kv_caches[layer_name]
            assert kv_layer.dim() >= 4 and kv_layer.shape[0] == 2, (
                f"C2C: unexpected KV layout {tuple(kv_layer.shape)}"
            )
            num_blocks, block_size = kv_layer.shape[1], kv_layer.shape[2]
            flat = kv_layer.reshape(2, num_blocks * block_size, *kv_layer.shape[3:])
            # [N, H, D] -> [1, H, N, D]
            tgt_k = flat[0, slots].permute(1, 0, 2).unsqueeze(0)
            tgt_v = flat[1, slots].permute(1, 0, 2).unsqueeze(0)
            src_k = staged[src_layer, 0, positions].permute(1, 0, 2).unsqueeze(0)
            src_v = staged[src_layer, 1, positions].permute(1, 0, 2).unsqueeze(0)
            fused_k, fused_v = proj((src_k, src_v), (tgt_k, tgt_v))
            flat[0, slots] = fused_k.squeeze(0).permute(1, 0, 2).to(flat.dtype)
            flat[1, slots] = fused_v.squeeze(0).permute(1, 0, 2).to(flat.dtype)
        end_evt.record()
        end_evt.synchronize()
        fusion_ms = start_evt.elapsed_time(end_evt)
        self._fused_ranges[req.key] = end
        timing = self._timings.get(req.key)
        if timing is not None:
            timing["fusion_ms"] = fusion_ms
            timing["fused_tokens"] = int(end - start)
            out_path = os.path.join(self._dir, f"timing_{req.key}.json")
            with open(out_path, "w") as f:
                json.dump(timing, f, indent=2)
        logger.info(
            "C2C consumer: fused %d tokens (pos %d..%d) across %d layers "
            "in %.2f ms key=%s",
            end - start,
            start,
            end - 1,
            len(layer_names),
            fusion_ms,
            req.key,
        )

    def wait_for_save(self) -> None:
        if not self._is_producer:
            jobs, self._fuse_jobs = self._fuse_jobs, []
            for req in jobs:
                self._run_fusion(req)
            return
        for key, layers in list(self._pending_layers.items()):
            t0 = time.perf_counter()
            stacked = torch.stack([t for _, t in layers], dim=0).contiguous()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            ipc = reduce_tensor(stacked)
            self._staged[key] = stacked
            manifest = {
                "key": key,
                "layer_names": [n for n, _ in layers],
                "shape": list(stacked.shape),
                "dtype": str(stacked.dtype),
                "ipc": ipc,
                "extract_ms": (t1 - t0) * 1000.0,
            }
            tmp = os.path.join(self._dir, f"{key}.manifest.tmp")
            final = os.path.join(self._dir, f"{key}.manifest")
            with open(tmp, "wb") as f:
                pickle.dump(manifest, f)
            os.replace(tmp, final)
            nbytes = stacked.numel() * stacked.element_size()
            logger.info(
                "C2C producer: staged %.2f MB for key=%s (%d layers, shape=%s)",
                nbytes / 1e6,
                key,
                len(layers),
                tuple(stacked.shape),
            )
            del self._pending_layers[key]
            # Bound staging memory: keep only the 8 most recent keys.
            while len(self._staged) > 8:
                oldest = next(iter(self._staged))
                del self._staged[oldest]
                p = os.path.join(self._dir, f"{oldest}.manifest")
                if os.path.exists(p):
                    os.remove(p)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        return None, None
