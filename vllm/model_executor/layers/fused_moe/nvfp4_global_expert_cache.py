# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-wide predictor-driven NVFP4 expert cache for single-GPU decode.

The cache retains packed Marlin weights and their scales in a shared GPU slot
bank. Expert identity is ``(layer_index, expert_id)``. Native routing remains
authoritative: trained predictions schedule asynchronous copies, while a miss
loads the router-selected expert before the fused MoE kernel executes.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.expert_weight_cache import (
        ExpertWeightResult,
    )
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

logger = init_logger(__name__)

EXPERT_PARAMETER_NAMES = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale",
    "w2_weight_scale",
    "w13_weight_scale_2",
    "w2_weight_scale_2",
)


class CacheEntry:
    def __init__(
        self,
        *,
        slot: int,
        last_access: int,
        speculative: bool,
    ) -> None:
        self.slot = slot
        self.last_access = last_access
        self.speculative = speculative


class GlobalNVFP4ExpertCache:
    def __init__(
        self,
        *,
        layers: tuple[RoutedExperts, ...],
        capacity: int,
        predictor_checkpoint: str,
        history_tokens: int,
        metrics_path: str,
    ) -> None:
        if capacity < 8:
            raise ValueError(
                "model-wide expert cache capacity must fit MiniMax top-k=8"
            )
        if not layers:
            raise ValueError("at least one routed-expert layer is required")
        self.layers = layers
        self.num_layers = len(layers)
        self.num_experts = int(layers[0].w13_weight.shape[0])
        self.capacity = min(capacity, self.num_layers * self.num_experts)
        self.history_tokens = history_tokens
        self.device = layers[0].w13_weight.device
        if self.device.type != "cuda":
            raise ValueError("NVFP4 cache sources must be CUDA or CUDA UVA views")

        self.sources: tuple[dict[str, torch.Tensor], ...] = tuple(
            {
                "w13_weight": layer.w13_weight,
                "w2_weight": layer.w2_weight,
                "w13_weight_scale": layer.w13_weight_scale,
                "w2_weight_scale": layer.w2_weight_scale,
                "w13_weight_scale_2": layer.w13_weight_scale_2,
                "w2_weight_scale_2": layer.w2_weight_scale_2,
            }
            for layer in layers
        )
        reference = self.sources[0]
        for layer_index, source in enumerate(self.sources):
            for name in EXPERT_PARAMETER_NAMES:
                if source[name].shape[0] != self.num_experts:
                    raise ValueError(
                        f"layer {layer_index} {name} lacks an expert dimension"
                    )
                if source[name].shape[1:] != reference[name].shape[1:]:
                    raise ValueError(
                        f"layer {layer_index} {name} shape differs from layer 0"
                    )

        self.reference = reference
        self.buffers = self.allocate_buffers(self.capacity)
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.slot_events: list[torch.cuda.Event | None] = [None] * self.capacity
        self.entries: dict[tuple[int, int], CacheEntry] = {}
        self.slot_keys: list[tuple[int, int] | None] = [None] * self.capacity
        self.free_slots = list(reversed(range(self.capacity)))
        self.clock = 0

        self.predictor = self.load_predictor(predictor_checkpoint)
        self.metrics_path = Path(metrics_path) if metrics_path else None
        self.capacity_control_path = (
            self.metrics_path.with_suffix(self.metrics_path.suffix + ".capacity")
            if self.metrics_path is not None
            else None
        )
        if self.metrics_path is not None:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

        self.current_input_ids: list[int] = []
        self.current_routes: list[torch.Tensor | None] = [None] * self.num_layers
        self.history_input_ids: list[int] = []
        self.history_routes: list[torch.Tensor] = []
        self.predicted_routes: torch.Tensor | None = None
        self.prefetch_window_layers = min(
            self.num_layers,
            max(1, self.capacity // 16),
        )

        self.hits = 0
        self.demand_misses = 0
        self.speculative_loads = 0
        self.useful_speculations = 0
        self.evictions = 0
        self.bytes_copied = 0
        self.predictor_calls = 0
        self.predictor_seconds = 0.0
        self.processed_tokens = 0
        self.max_resident = 0
        self.resize_count = 0
        self.write_metrics()

        logger.info(
            "Enabled global predictor-driven NVFP4 cache: %d/%d layer-expert "
            "slots (%.3f%%), %.3f GiB GPU, predictor=%s",
            self.capacity,
            self.num_layers * self.num_experts,
            100.0 * self.capacity / (self.num_layers * self.num_experts),
            self.gpu_bytes / 1024**3,
            predictor_checkpoint,
        )

    def migrate_all_sources_to_uva(self) -> None:
        """Move every full expert tensor to host memory after weight loading.

        vLLM's construction-time offload limit cannot cover the whole MiniMax
        expert set on this host because the temporary, pre-quantized parameter
        layout is substantially larger than the loaded NVFP4 checkpoint. At
        this point weight loading and Marlin processing are complete, so the
        tensors have their final packed size and fit in host memory. Replacing
        every source with a UVA view ensures that the only expert weights held
        in physical H100 memory are the explicitly capacity-limited cache bank.
        """
        source_bytes = 0
        for layer in self.layers:
            parameters: tuple[torch.Tensor, ...] = (
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                layer.w13_weight_scale_2,
                layer.w2_weight_scale_2,
            )
            for parameter in parameters:
                if parameter._vllm_is_uva_offloaded:
                    source_bytes += parameter.numel() * parameter.element_size()
                    continue
                cpu_data = torch.empty(
                    parameter.shape,
                    dtype=parameter.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                cpu_data.copy_(parameter.data)
                parameter.data = get_accelerator_view_from_cpu_tensor(cpu_data)
                parameter._vllm_is_uva_offloaded = True
                source_bytes += parameter.numel() * parameter.element_size()
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        logger.info(
            "Migrated all packed NVFP4 expert sources to UVA host memory: %.3f GiB",
            source_bytes / 1024**3,
        )

    @property
    def expert_bytes(self) -> int:
        return (
            sum(
                source.numel() * source.element_size()
                for source in self.sources[0].values()
            )
            // self.num_experts
        )

    @property
    def gpu_bytes(self) -> int:
        return sum(
            buffer.numel() * buffer.element_size() for buffer in self.buffers.values()
        )

    def allocate_buffers(self, capacity: int) -> dict[str, torch.Tensor]:
        return {
            name: torch.empty(
                (capacity, *self.reference[name].shape[1:]),
                dtype=self.reference[name].dtype,
                device=self.device,
            )
            for name in EXPERT_PARAMETER_NAMES
        }

    def resize(self, capacity: int) -> None:
        selected = min(capacity, self.num_layers * self.num_experts)
        if selected < 8:
            raise ValueError("resized cache capacity must fit MiniMax top-k=8")
        if selected == self.capacity:
            return
        torch.cuda.synchronize(self.device)
        old_buffers = self.buffers
        self.buffers = {}
        self.entries.clear()
        self.slot_keys.clear()
        self.free_slots.clear()
        self.slot_events.clear()
        del old_buffers
        torch.cuda.empty_cache()
        self.capacity = selected
        self.buffers = self.allocate_buffers(self.capacity)
        self.slot_events = [None] * self.capacity
        self.slot_keys = [None] * self.capacity
        self.free_slots = list(reversed(range(self.capacity)))
        self.prefetch_window_layers = min(
            self.num_layers,
            max(1, self.capacity // 16),
        )
        self.predicted_routes = None
        self.resize_count += 1
        self.write_metrics()
        logger.info(
            "Resized global NVFP4 expert cache to %d/%d slots (%.3f%%, %.3f GiB)",
            self.capacity,
            self.num_layers * self.num_experts,
            100.0 * self.capacity / (self.num_layers * self.num_experts),
            self.gpu_bytes / 1024**3,
        )

    def apply_requested_capacity(self) -> None:
        if (
            self.capacity_control_path is None
            or not self.capacity_control_path.exists()
        ):
            return
        requested = int(self.capacity_control_path.read_text().strip())
        self.resize(requested)

    def load_predictor(self, checkpoint_path: str) -> nn.Module:
        from src.routing_prediction.models import (
            RouteHistoryModelSpec,
            build_route_history_model,
        )

        checkpoint: dict[str, object] = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        raw_config = checkpoint["config"]
        if not isinstance(raw_config, dict):
            raise TypeError("predictor checkpoint config must be a dictionary")
        training = raw_config["training"]
        model_config = raw_config["model"]
        data_config = raw_config["data"]
        if not isinstance(training, dict):
            raise TypeError("predictor training config must be a dictionary")
        if not isinstance(model_config, dict):
            raise TypeError("predictor model config must be a dictionary")
        if not isinstance(data_config, dict):
            raise TypeError("predictor data config must be a dictionary")
        if training["experiment"] != "cache_history_transformer":
            raise ValueError("checkpoint is not a cache-history predictor")
        if int(model_config["num_layers"]) != self.num_layers:
            raise ValueError("predictor and served model layer counts differ")
        if int(model_config["num_experts"]) != self.num_experts:
            raise ValueError("predictor and served model expert counts differ")
        cache_horizons = model_config["cache_horizons"]
        if not isinstance(cache_horizons, list | tuple):
            raise TypeError("cache_horizons must be a list or tuple")
        spec = RouteHistoryModelSpec(
            vocab_size=int(model_config["vocab_size"]),
            num_layers=int(model_config["num_layers"]),
            num_experts=int(model_config["num_experts"]),
            top_k=int(model_config["top_k"]),
            width=int(model_config["history_width"]),
            transformer_layers=int(model_config["history_layers"]),
            heads=int(model_config["history_heads"]),
            ffn_width=int(model_config["history_ffn_width"]),
            dropout=float(model_config["history_dropout"]),
            max_sequence_length=int(data_config["history_tokens"]),
            hidden_feature_size=int(model_config["hidden_feature_size"]),
            output_horizons=len(cache_horizons),
        )
        model = build_route_history_model(spec)
        state = checkpoint["model_state_dict"]
        if not isinstance(state, dict):
            raise TypeError("predictor model_state_dict must be a dictionary")
        model.load_state_dict(state)
        model.eval().to(device=self.device, dtype=torch.bfloat16)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def reset_history(self) -> None:
        self.history_input_ids.clear()
        self.history_routes.clear()
        self.predicted_routes = None

    def begin_step(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> None:
        if input_ids is None:
            raise ValueError("predictor-driven expert caching requires input_ids")
        token_ids = input_ids.detach().reshape(-1).to(device="cpu").tolist()
        position_ids = positions.detach().reshape(-1).to(device="cpu").tolist()
        if len(token_ids) != len(position_ids):
            raise ValueError("input_ids and positions have different lengths")
        if position_ids and int(position_ids[0]) == 0:
            self.apply_requested_capacity()
            self.reset_history()
        self.current_input_ids = [int(token_id) for token_id in token_ids]
        self.current_routes = [None] * self.num_layers

    def victim(self, protected: frozenset[tuple[int, int]]) -> tuple[int, int]:
        candidates = self.entries.keys() - protected
        if not candidates:
            raise RuntimeError("no evictable global expert-cache slot remains")
        return min(
            candidates,
            key=lambda key: (self.entries[key].last_access, key),
        )

    def resolve(
        self,
        key: tuple[int, int],
        *,
        speculative: bool,
        protected: frozenset[tuple[int, int]],
    ) -> tuple[int, bool]:
        self.clock += 1
        entry = self.entries.get(key)
        if entry is not None:
            if not speculative:
                self.hits += 1
                if entry.speculative:
                    self.useful_speculations += 1
                    entry.speculative = False
            entry.last_access = self.clock
            return entry.slot, False

        if self.free_slots:
            slot = self.free_slots.pop()
        else:
            victim = self.victim(protected)
            victim_entry = self.entries.pop(victim)
            slot = victim_entry.slot
            self.slot_keys[slot] = None
            self.evictions += 1
        self.entries[key] = CacheEntry(
            slot=slot,
            last_access=self.clock,
            speculative=speculative,
        )
        self.slot_keys[slot] = key
        if speculative:
            self.speculative_loads += 1
        else:
            self.demand_misses += 1
        self.max_resident = max(self.max_resident, len(self.entries))
        return slot, True

    def schedule_copies(
        self,
        copies: list[tuple[tuple[int, int], int]],
    ) -> None:
        if not copies:
            return
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            for (layer_index, expert_id), slot in copies:
                source = self.sources[layer_index]
                for name in EXPERT_PARAMETER_NAMES:
                    self.buffers[name][slot].copy_(
                        source[name][expert_id],
                        non_blocking=True,
                    )
                event = torch.cuda.Event()
                event.record(self.copy_stream)
                self.slot_events[slot] = event
                self.bytes_copied += self.expert_bytes

    def demand(
        self,
        layer_index: int,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        unique_ids = tuple(
            sorted(
                {
                    int(expert_id)
                    for expert_id in topk_ids.detach().reshape(-1).to("cpu").tolist()
                }
            )
        )
        protected = frozenset((layer_index, expert_id) for expert_id in unique_ids)
        copies: list[tuple[tuple[int, int], int]] = []
        slots: dict[int, int] = {}
        for expert_id in unique_ids:
            key = (layer_index, expert_id)
            slot, missing = self.resolve(
                key,
                speculative=False,
                protected=protected,
            )
            slots[expert_id] = slot
            if missing:
                copies.append((key, slot))
        self.schedule_copies(copies)
        current_stream = torch.cuda.current_stream(self.device)
        for slot in slots.values():
            event = self.slot_events[slot]
            if event is not None:
                current_stream.wait_event(event)
        remapped = torch.empty_like(topk_ids)
        for expert_id, slot in slots.items():
            remapped.masked_fill_(topk_ids == expert_id, slot)
        return remapped, unique_ids

    def prefetch_layer(self, layer_index: int) -> None:
        if self.predicted_routes is None or layer_index >= self.num_layers:
            return
        predicted = tuple(
            int(expert_id) for expert_id in self.predicted_routes[layer_index].tolist()
        )
        protected = frozenset()
        copies: list[tuple[tuple[int, int], int]] = []
        for expert_id in predicted:
            key = (layer_index, expert_id)
            slot, missing = self.resolve(
                key,
                speculative=True,
                protected=protected,
            )
            if missing:
                copies.append((key, slot))
        self.schedule_copies(copies)

    def release_layer(self, layer_index: int) -> None:
        keys = [key for key in self.entries if key[0] == layer_index]
        for key in keys:
            entry = self.entries.pop(key)
            self.slot_keys[entry.slot] = None
            self.free_slots.append(entry.slot)

    def prepare(
        self,
        layer_index: int,
        topk_ids: torch.Tensor,
    ) -> ExpertWeightResult:
        from vllm.model_executor.layers.fused_moe.expert_weight_cache import (
            ExpertWeightResult,
        )

        self.current_routes[layer_index] = topk_ids.detach().to(
            device="cpu",
            dtype=torch.int64,
        )
        source = self.sources[layer_index]
        if topk_ids.shape[0] != 1:
            return ExpertWeightResult(
                w13_weight=source["w13_weight"],
                w2_weight=source["w2_weight"],
                topk_ids=topk_ids,
                global_num_experts=self.num_experts,
            )

        remapped, _ = self.demand(layer_index, topk_ids)
        return ExpertWeightResult(
            w13_weight=self.buffers["w13_weight"],
            w2_weight=self.buffers["w2_weight"],
            topk_ids=remapped,
            w13_scale=self.buffers["w13_weight_scale"],
            w2_scale=self.buffers["w2_weight_scale"],
            w13_scale_2=self.buffers["w13_weight_scale_2"],
            w2_scale_2=self.buffers["w2_weight_scale_2"],
            global_num_experts=self.capacity,
        )

    def finish_layer(self, layer_index: int) -> None:
        if self.predicted_routes is not None and self.capacity < self.num_layers * 16:
            self.release_layer(layer_index)
            self.prefetch_layer(layer_index + self.prefetch_window_layers)
        if layer_index != self.num_layers - 1:
            return
        routes = self.current_routes
        if any(route is None for route in routes):
            raise RuntimeError("not every MiniMax layer recorded routed experts")
        concrete_routes = [route for route in routes if route is not None]
        rows = concrete_routes[0].shape[0]
        if rows != len(self.current_input_ids):
            raise ValueError("routed rows do not match current input token count")
        if any(route.shape != (rows, 8) for route in concrete_routes):
            raise ValueError("MiniMax routed experts must have shape [tokens, 8]")
        for row in range(rows):
            token_routes = torch.stack(
                [route[row] for route in concrete_routes],
                dim=0,
            )
            self.history_input_ids.append(self.current_input_ids[row])
            self.history_routes.append(token_routes)
        self.history_input_ids = self.history_input_ids[-self.history_tokens :]
        self.history_routes = self.history_routes[-self.history_tokens :]
        self.processed_tokens += rows
        self.predict_and_prefetch()
        if self.processed_tokens % 16 == 0 or rows > 1:
            self.write_metrics()

    def predict_and_prefetch(self) -> None:
        if not self.history_input_ids:
            return
        input_ids = torch.tensor(
            self.history_input_ids,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        route_history = (
            torch.stack(self.history_routes, dim=0)
            .to(
                device=self.device,
                non_blocking=True,
            )
            .unsqueeze(0)
        )
        started = time.perf_counter()
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ),
        ):
            logits = self.predictor(input_ids, route_history)
            selected = torch.topk(
                logits[0, -1, 0].float(),
                16,
                dim=-1,
            ).indices
        self.predicted_routes = selected.to(device="cpu", dtype=torch.int64)
        self.predictor_seconds += time.perf_counter() - started
        self.predictor_calls += 1
        for layer_index in range(self.prefetch_window_layers):
            self.prefetch_layer(layer_index)

    def metrics(self) -> dict[str, int | float | str]:
        demand_accesses = self.hits + self.demand_misses
        return {
            "schema_version": 1,
            "mode": "global_nvfp4_predictor_cache_with_demand_fallback",
            "capacity": self.capacity,
            "capacity_fraction": self.capacity / (self.num_layers * self.num_experts),
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "resident": len(self.entries),
            "max_resident": self.max_resident,
            "expert_bytes": self.expert_bytes,
            "gpu_bytes": self.gpu_bytes,
            "hits": self.hits,
            "demand_misses": self.demand_misses,
            "hit_rate": self.hits / demand_accesses if demand_accesses else 0.0,
            "speculative_loads": self.speculative_loads,
            "useful_speculations": self.useful_speculations,
            "speculation_precision": (
                self.useful_speculations / self.speculative_loads
                if self.speculative_loads
                else 0.0
            ),
            "evictions": self.evictions,
            "bytes_copied": self.bytes_copied,
            "predictor_calls": self.predictor_calls,
            "predictor_seconds": self.predictor_seconds,
            "predictor_mean_ms": (
                1000.0 * self.predictor_seconds / self.predictor_calls
                if self.predictor_calls
                else 0.0
            ),
            "processed_tokens": self.processed_tokens,
            "history_tokens": len(self.history_input_ids),
            "prefetch_window_layers": self.prefetch_window_layers,
            "resize_count": self.resize_count,
        }

    def write_metrics(self) -> None:
        if self.metrics_path is None:
            return
        temporary = self.metrics_path.with_suffix(self.metrics_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.metrics(), indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, self.metrics_path)


GLOBAL_NVFP4_EXPERT_CACHE: GlobalNVFP4ExpertCache | None = None


def initialize_global_nvfp4_expert_cache(
    layers: tuple[RoutedExperts, ...],
    *,
    capacity: int,
    predictor_checkpoint: str,
    history_tokens: int,
    metrics_path: str,
) -> None:
    global GLOBAL_NVFP4_EXPERT_CACHE
    if GLOBAL_NVFP4_EXPERT_CACHE is not None:
        existing_layers = GLOBAL_NVFP4_EXPERT_CACHE.layers
        if len(existing_layers) == len(layers) and all(
            existing is requested
            for existing, requested in zip(existing_layers, layers, strict=True)
        ):
            logger.info("Global NVFP4 expert cache is already linked to this model")
            return
        raise RuntimeError("global NVFP4 expert cache was initialized twice")
    GLOBAL_NVFP4_EXPERT_CACHE = GlobalNVFP4ExpertCache(
        layers=layers,
        capacity=capacity,
        predictor_checkpoint=predictor_checkpoint,
        history_tokens=history_tokens,
        metrics_path=metrics_path,
    )


def begin_global_nvfp4_expert_cache_step(
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
) -> None:
    cache = GLOBAL_NVFP4_EXPERT_CACHE
    if cache is not None:
        cache.begin_step(input_ids, positions)


def prepare_global_nvfp4_expert_weights(
    layer_index: int,
    topk_ids: torch.Tensor,
) -> ExpertWeightResult:
    cache = GLOBAL_NVFP4_EXPERT_CACHE
    if cache is None:
        raise RuntimeError("global NVFP4 expert cache is not initialized")
    return cache.prepare(layer_index, topk_ids)


def finish_global_nvfp4_expert_layer(layer_index: int) -> None:
    cache = GLOBAL_NVFP4_EXPERT_CACHE
    if cache is None:
        raise RuntimeError("global NVFP4 expert cache is not initialized")
    cache.finish_layer(layer_index)
