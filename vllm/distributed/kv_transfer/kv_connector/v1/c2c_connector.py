# SPDX-License-Identifier: Apache-2.0
"""C2C (Cache-to-Cache) KV transfer connector prototype.

Phase A: measure raw cross-server KV cache transfer speed on the same GPU.

Producer (sharer model server): after prefill, extracts the prompt's KV from
the paged cache for every attention layer, stacks it into one staging tensor,
and publishes a CUDA IPC handle keyed by a hash of the prompt token ids.

Consumer (receiver model server): at the start of its own prefill forward,
looks up the manifest for the same prompt key, rebuilds the producer's staging
tensor via CUDA IPC, and copies it into its own GPU buffer, recording
open/copy timings to a JSON file for the benchmark harness.

This phase does NOT yet project/fuse the sharer KV into the receiver's paged
cache (that is Phase B, which adds the C2C projector from thu-nics/C2C).
"""

import json
import os
import pickle
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
    token_ids: list[int]
    slot_mapping: torch.Tensor
    is_store: bool


@dataclass
class C2CConnectorMetadata(KVConnectorMetadata):
    requests: list[C2CReqMeta] = field(default_factory=list)


class C2CConnector(KVConnectorBase_V1):
    """Prototype connector for C2C-style cross-model KV transfer."""

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
        os.makedirs(self._dir, exist_ok=True)
        # producer worker state: key -> staging tensor (kept alive for IPC)
        self._staged: dict[str, torch.Tensor] = {}
        # per-step accumulation: key -> list[(layer_name, kv_tensor)]
        self._pending_layers: dict[str, list[tuple[str, torch.Tensor]]] = {}
        # consumer worker state: keys already transferred
        self._transferred: set[str] = set()
        logger.info(
            "C2CConnector role=%s kv_role=%s dir=%s",
            role,
            self._kv_transfer_config.kv_role,
            self._dir,
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
            # Phase A: only handle single-step (non-chunked) prefills.
            if num_scheduled < len(token_ids):
                logger.warning(
                    "C2C: skipping chunked prefill request %s (%d < %d)",
                    new_req.req_id,
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
                    token_ids=token_ids,
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

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        if self._is_producer:
            return
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, C2CConnectorMetadata)
        for req in metadata.requests:
            if req.is_store or req.key in self._transferred:
                continue
            manifest_path = os.path.join(self._dir, f"{req.key}.manifest")
            if not os.path.exists(manifest_path):
                logger.info("C2C consumer: no manifest for key=%s", req.key)
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
            timing = {
                "key": req.key,
                "consumer_req_id": req.req_id,
                "num_prompt_tokens_consumer": len(req.token_ids),
                "producer_shape": list(local.shape),
                "dtype": str(local.dtype),
                "bytes": nbytes,
                "manifest_open_ms": (t1 - t0) * 1000.0,
                "ipc_copy_ms": copy_ms,
                "bandwidth_gb_s": (nbytes / 1e9) / (copy_ms / 1000.0)
                if copy_ms > 0
                else None,
                "producer_extract_ms": manifest["extract_ms"],
                "producer_publish_ms": manifest["publish_ms"],
            }
            self._transferred.add(req.key)
            # keep the transferred copy alive for Phase B fusion experiments
            self._staged[req.key] = local
            out_path = os.path.join(self._dir, f"timing_{req.key}.json")
            with open(out_path, "w") as f:
                json.dump(timing, f, indent=2)
            logger.info(
                "C2C consumer: transferred %.2f MB in %.3f ms (%.1f GB/s) key=%s",
                nbytes / 1e6,
                copy_ms,
                timing["bandwidth_gb_s"] or -1.0,
                req.key,
            )

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

    def wait_for_save(self) -> None:
        if not self._is_producer:
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
                "publish_ms": 0.0,
            }
            t2 = time.perf_counter()
            tmp = os.path.join(self._dir, f"{key}.manifest.tmp")
            final = os.path.join(self._dir, f"{key}.manifest")
            manifest["publish_ms"] = (time.perf_counter() - t2) * 1000.0
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
                for suffix in (".manifest",):
                    p = os.path.join(self._dir, f"{oldest}{suffix}")
                    if os.path.exists(p):
                        os.remove(p)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        return None, None
