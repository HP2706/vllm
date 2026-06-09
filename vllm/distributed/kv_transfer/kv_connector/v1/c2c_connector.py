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
import io
import os
import pickle
import re
import socket
import struct
import threading
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
        self._transport: str = self._kv_transfer_config.get_from_extra_config(
            "c2c_transport", "ipc_manifest"
        )
        self._dir: str = self._kv_transfer_config.get_from_extra_config(
            "c2c_dir", "/dev/shm/c2c_kv"
        )
        self._timing_dir: str = self._kv_transfer_config.get_from_extra_config(
            "c2c_timing_dir", self._dir
        )
        self._tcp_host: str = self._kv_transfer_config.get_from_extra_config(
            "c2c_host", "127.0.0.1"
        )
        self._tcp_bind_host: str = self._kv_transfer_config.get_from_extra_config(
            "c2c_bind_host", self._tcp_host
        )
        self._tcp_port: int = int(
            self._kv_transfer_config.get_from_extra_config("c2c_port", 8077)
        )
        self._tcp_wait_s: float = float(
            self._kv_transfer_config.get_from_extra_config("c2c_tcp_wait_s", 30.0)
        )
        self._checkpoints_dir: str | None = self._kv_transfer_config.get_from_extra_config(
            "c2c_checkpoints_dir", None
        )
        if self._transport == "ipc_manifest":
            os.makedirs(self._dir, exist_ok=True)
        os.makedirs(self._timing_dir, exist_ok=True)
        # producer worker state: key -> staging tensor (kept alive for IPC)
        self._staged: dict[str, torch.Tensor] = {}
        # per-step accumulation: key -> list[(layer_name, kv_tensor)]
        self._pending_layers: dict[str, list[tuple[str, torch.Tensor]]] = {}
        # consumer worker state
        self._transferred: set[str] = set()
        self._fused_ranges: dict[tuple[str, int], int] = {}
        self._fuse_jobs: list[C2CReqMeta] = []
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._projectors: list[torch.nn.Module] | None = None
        self._layer_mapping: dict[int, tuple[int, int]] = {}
        self._timings: dict[str, dict[str, Any]] = {}
        self._tcp_server_thread: threading.Thread | None = None
        if (
            role == KVConnectorRole.WORKER
            and self._is_producer
            and self._transport == "tcp"
        ):
            self._start_tcp_server()
        logger.info(
            "C2CConnector role=%s kv_role=%s transport=%s dir=%s fuse=%s",
            role,
            self._kv_transfer_config.kv_role,
            self._transport,
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

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()
        self._fuse_jobs = []

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._kv_caches = kv_caches

    def _ordered_layer_names(self) -> list[str]:
        def layer_idx(name: str) -> int:
            m = re.search(r"layers\.(\d+)\.", name)
            assert m is not None, f"C2C: cannot parse layer index from {name}"
            return int(m.group(1))

        return sorted(self._kv_caches.keys(), key=layer_idx)

    @staticmethod
    def _recv_exact(conn: socket.socket, nbytes: int) -> bytes:
        chunks: list[bytes] = []
        remaining = nbytes
        while remaining > 0:
            chunk = conn.recv(remaining)
            assert chunk, "C2C TCP connection closed before payload completed"
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _start_tcp_server(self) -> None:
        def serve() -> None:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((self._tcp_bind_host, self._tcp_port))
                server.listen(64)
                logger.info(
                    "C2C producer TCP server listening on %s:%d",
                    self._tcp_bind_host,
                    self._tcp_port,
                )
                while True:
                    conn, _ = server.accept()
                    with conn:
                        raw_len = self._recv_exact(conn, 4)
                        req_len = struct.unpack("!I", raw_len)[0]
                        raw_req = self._recv_exact(conn, req_len)
                        req = json.loads(raw_req.decode("utf-8"))
                        key = req["key"]
                        deadline = time.perf_counter() + self._tcp_wait_s
                        while key not in self._staged:
                            assert time.perf_counter() < deadline, (
                                f"C2C producer timed out waiting for key={key}"
                            )
                            time.sleep(0.001)
                        cpu_tensor = self._staged[key].detach().cpu()
                        buf = io.BytesIO()
                        torch.save(cpu_tensor, buf)
                        payload = buf.getvalue()
                        conn.sendall(struct.pack("!Q", len(payload)))
                        conn.sendall(payload)

        self._tcp_server_thread = threading.Thread(
            target=serve,
            name="c2c-producer-tcp",
            daemon=True,
        )
        self._tcp_server_thread.start()

    def _fetch_tcp(self, key: str, device: torch.device) -> torch.Tensor:
        req = json.dumps({"key": key}).encode("utf-8")
        t0 = time.perf_counter()
        with socket.create_connection(
            (self._tcp_host, self._tcp_port), timeout=self._tcp_wait_s
        ) as conn:
            conn.sendall(struct.pack("!I", len(req)))
            conn.sendall(req)
            raw_len = self._recv_exact(conn, 8)
            payload_len = struct.unpack("!Q", raw_len)[0]
            payload = self._recv_exact(conn, payload_len)
        t1 = time.perf_counter()
        cpu_tensor = torch.load(io.BytesIO(payload), map_location="cpu")
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        local = cpu_tensor.to(device=device, non_blocking=False)
        end_evt.record()
        end_evt.synchronize()
        copy_ms = start_evt.elapsed_time(end_evt)
        self._timings[key] = {
            "key": key,
            "producer_shape": list(local.shape),
            "bytes": local.numel() * local.element_size(),
            "transport": "tcp",
            "tcp_fetch_ms": (t1 - t0) * 1000.0,
            "h2d_copy_ms": copy_ms,
            "fusion_ms": None,
        }
        return local

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
        assert self._kv_caches, "C2C consumer must register KV caches before loading"
        device = next(iter(self._kv_caches.values())).device
        for req in metadata.requests:
            if req.is_store:
                continue
            if req.key not in self._transferred:
                if self._transport == "tcp":
                    local = self._fetch_tcp(req.key, device)
                    nbytes = local.numel() * local.element_size()
                else:
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
                    self._timings[req.key] = {
                        "key": req.key,
                        "num_prompt_tokens_consumer": req.num_prompt_tokens,
                        "producer_shape": list(local.shape),
                        "bytes": nbytes,
                        "transport": "ipc_manifest",
                        "manifest_open_ms": (t1 - t0) * 1000.0,
                        "ipc_copy_ms": copy_ms,
                        "bandwidth_gb_s": (nbytes / 1e9) / (copy_ms / 1000.0)
                        if copy_ms > 0
                        else None,
                        "producer_extract_ms": manifest["extract_ms"],
                        "fusion_ms": None,
                    }
                self._transferred.add(req.key)
                self._staged[req.key] = local
                if self._checkpoints_dir is None:
                    out_path = os.path.join(self._timing_dir, f"timing_{req.key}.json")
                    with open(out_path, "w") as f:
                        json.dump(self._timings[req.key], f, indent=2)
                logger.info(
                    "C2C consumer: transferred %.2f MB for key=%s via %s",
                    nbytes / 1e6,
                    req.key,
                    self._transport,
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
    def _run_fusion_layer(
        self,
        req: C2CReqMeta,
        layer_name: str,
        kv_layer: torch.Tensor,
    ) -> None:
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
        layer_names = self._ordered_layer_names()
        tgt_layer = layer_names.index(layer_name)
        already = self._fused_ranges.get((req.key, tgt_layer), 0)
        start = max(req.num_computed_tokens, already)
        end = req.num_prompt_tokens - 1
        if start >= end:
            return
        positions = torch.arange(start, end, device=device)
        slots = req.slot_mapping.to(device)[start:end]

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        assert len(layer_names) == len(self._layer_mapping), (
            f"C2C: receiver has {len(layer_names)} attention layers but "
            f"mapping covers {len(self._layer_mapping)}"
        )
        src_layer, proj_idx = self._layer_mapping[tgt_layer]
        proj = self._projectors[proj_idx]
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
        self._fused_ranges[(req.key, tgt_layer)] = end
        timing = self._timings.get(req.key)
        if timing is not None:
            timing["fusion_ms"] = float(timing["fusion_ms"] or 0.0) + fusion_ms
            timing["fused_tokens"] = int(end - start)
            out_path = os.path.join(self._timing_dir, f"timing_{req.key}.json")
            with open(out_path, "w") as f:
                json.dump(timing, f, indent=2)
        logger.info(
            "C2C consumer: fused %d tokens (pos %d..%d) for layer %s "
            "in %.2f ms key=%s",
            end - start,
            start,
            end - 1,
            layer_name,
            fusion_ms,
            req.key,
        )

    def mutate_kv_post_write(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
    ) -> None:
        if self._is_producer or self._checkpoints_dir is None:
            return
        jobs = list(self._fuse_jobs)
        for req in jobs:
            if req.key in self._staged:
                self._run_fusion_layer(req, layer_name, kv_layer)

    def wait_for_save(self) -> None:
        if not self._is_producer:
            return
        for key, layers in list(self._pending_layers.items()):
            t0 = time.perf_counter()
            stacked = torch.stack([t for _, t in layers], dim=0).contiguous()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            self._staged[key] = stacked
            if self._transport == "ipc_manifest":
                ipc = reduce_tensor(stacked)
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
                "C2C producer: staged %.2f MB for key=%s via %s "
                "(%d layers, shape=%s, extract=%.2f ms)",
                nbytes / 1e6,
                key,
                self._transport,
                len(layers),
                tuple(stacked.shape),
                (t1 - t0) * 1000.0,
            )
            del self._pending_layers[key]
            # Bound staging memory: keep only the 8 most recent keys.
            while len(self._staged) > 8:
                oldest = next(iter(self._staged))
                del self._staged[oldest]
                if self._transport == "ipc_manifest":
                    p = os.path.join(self._dir, f"{oldest}.manifest")
                    if os.path.exists(p):
                        os.remove(p)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        return None, None
