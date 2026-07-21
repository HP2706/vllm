# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental expert-granular CPU/GPU weight cache.

The first supported configuration is a single-GPU, eager, unquantized
``FusedMoE`` layer. The cache keeps complete expert tensors in pinned CPU
memory and exposes fixed-size GPU buffers to existing fused MoE kernels.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterable
from dataclasses import dataclass

import torch

_ACTIVE_CACHES: weakref.WeakSet[CachedExpertWeights] = weakref.WeakSet()


@dataclass(frozen=True)
class ExpertWeightResult:
    """GPU-resident weights and cache-slot-remapped router output."""

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    topk_ids: torch.Tensor


@dataclass
class CacheEntry:
    """Mutable LFRU state for one resident expert."""

    slot: int
    frequency: int
    last_access: int
    speculative: bool


@dataclass(frozen=True)
class CacheDecision:
    """Result of resolving one expert through the cache index."""

    expert_id: int
    slot: int
    hit: bool
    evicted_expert_id: int | None
    useful_speculation: bool


class LFRUCacheIndex:
    """Device-independent frequency-weighted LRU slot index."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.clock = 0
        self.entries: dict[int, CacheEntry] = {}
        self.free_slots = list(reversed(range(capacity)))

    def _victim(self, protected_expert_ids: frozenset[int]) -> int:
        if not self.entries:
            raise RuntimeError("cannot evict from an empty cache")

        candidates = self.entries.keys() - protected_expert_ids
        if not candidates:
            raise RuntimeError("no evictable cache slot remains")
        victim_id = min(
            candidates,
            key=lambda expert_id: (
                self.entries[expert_id].frequency
                / (self.clock - self.entries[expert_id].last_access + 1),
                self.entries[expert_id].last_access,
                expert_id,
            ),
        )
        return victim_id

    def resolve(
        self,
        expert_id: int,
        *,
        speculative: bool,
        protected_expert_ids: frozenset[int] = frozenset(),
    ) -> CacheDecision:
        self.clock += 1
        existing = self.entries.get(expert_id)
        if existing is not None:
            useful_speculation = not speculative and existing.speculative
            existing.frequency += 1
            existing.last_access = self.clock
            if not speculative:
                existing.speculative = False
            return CacheDecision(
                expert_id=expert_id,
                slot=existing.slot,
                hit=True,
                evicted_expert_id=None,
                useful_speculation=useful_speculation,
            )

        evicted_expert_id: int | None = None
        if self.free_slots:
            slot = self.free_slots.pop()
        else:
            evicted_expert_id = self._victim(protected_expert_ids)
            slot = self.entries.pop(evicted_expert_id).slot

        self.entries[expert_id] = CacheEntry(
            slot=slot,
            frequency=1,
            last_access=self.clock,
            speculative=speculative,
        )
        return CacheDecision(
            expert_id=expert_id,
            slot=slot,
            hit=False,
            evicted_expert_id=evicted_expert_id,
            useful_speculation=False,
        )

    def resident_experts(self) -> tuple[int, ...]:
        return tuple(sorted(self.entries))

    def admit_speculative(
        self, expert_ids: tuple[int, ...]
    ) -> tuple[tuple[int, ...], int]:
        """Admit predictions only into currently unused cache slots.

        A prediction must never evict demand-loaded state. Poor predictors can
        otherwise consume more PCIe bandwidth *and* lower the demand hit rate,
        which is strictly worse than not speculating.
        """
        missing_ids = tuple(
            expert_id for expert_id in expert_ids if expert_id not in self.entries
        )
        admitted_count = min(len(missing_ids), len(self.free_slots))
        return missing_ids[:admitted_count], len(missing_ids) - admitted_count


class CachedExpertWeights:
    """Pinned-CPU expert weights backed by fixed GPU cache slots.

    ``prefetch`` schedules non-blocking H2D copies and returns immediately.
    ``prepare`` makes the exact router-selected experts resident, waits only
    for their per-slot copy events, and returns IDs remapped to cache slots.
    """

    def __init__(
        self,
        capacity: int,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
    ) -> None:
        if w13_weight.shape[0] != w2_weight.shape[0]:
            raise ValueError(
                "w13_weight and w2_weight must have the same expert dimension"
            )
        if w13_weight.device.type != "cuda" or w2_weight.device.type != "cuda":
            raise ValueError("initial expert weights must be CUDA tensors")

        self.num_experts = int(w13_weight.shape[0])
        self.capacity = min(capacity, self.num_experts)
        self.index = LFRUCacheIndex(self.capacity)
        self.device = w13_weight.device

        self.cpu_w13_weight = self._copy_to_pinned_cpu(w13_weight)
        self.cpu_w2_weight = self._copy_to_pinned_cpu(w2_weight)
        self.gpu_w13_weight = torch.empty(
            (self.capacity, *w13_weight.shape[1:]),
            dtype=w13_weight.dtype,
            device=self.device,
        )
        self.gpu_w2_weight = torch.empty(
            (self.capacity, *w2_weight.shape[1:]),
            dtype=w2_weight.dtype,
            device=self.device,
        )
        self.expert_to_slot = torch.full(
            (self.num_experts,),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.slot_ready_events = [
            torch.cuda.Event(enable_timing=False) for _ in range(self.capacity)
        ]

        self.hits = 0
        self.demand_misses = 0
        self.speculative_loads = 0
        self.skipped_speculations = 0
        self.useful_speculations = 0
        self.evictions = 0
        self.bytes_copied = 0
        self.next_cache: CachedExpertWeights | None = None
        _ACTIVE_CACHES.add(self)

    @staticmethod
    def _copy_to_pinned_cpu(source: torch.Tensor) -> torch.Tensor:
        destination = torch.empty_strided(
            size=source.size(),
            stride=source.stride(),
            dtype=source.dtype,
            layout=source.layout,
            device="cpu",
            pin_memory=True,
        )
        destination.copy_(source)
        return destination

    @property
    def bytes_per_expert(self) -> int:
        w13_bytes = self.cpu_w13_weight[0].numel() * self.cpu_w13_weight.element_size()
        w2_bytes = self.cpu_w2_weight[0].numel() * self.cpu_w2_weight.element_size()
        return int(w13_bytes + w2_bytes)

    @property
    def cpu_bytes(self) -> int:
        return self.num_experts * self.bytes_per_expert

    @property
    def gpu_bytes(self) -> int:
        return self.capacity * self.bytes_per_expert

    def _validate_ids(self, expert_ids: Iterable[int]) -> tuple[int, ...]:
        unique_ids = tuple(sorted(set(expert_ids)))
        for expert_id in unique_ids:
            if expert_id < 0 or expert_id >= self.num_experts:
                raise ValueError(
                    f"expert ID {expert_id} is outside [0, {self.num_experts})"
                )
        if len(unique_ids) > self.capacity:
            raise RuntimeError(
                f"router selected {len(unique_ids)} unique experts, but the "
                f"expert cache has only {self.capacity} slots"
            )
        return unique_ids

    def _schedule(self, expert_ids: tuple[int, ...], *, speculative: bool) -> None:
        protected_expert_ids = frozenset(expert_ids)
        decisions = [
            self.index.resolve(
                expert_id,
                speculative=speculative,
                protected_expert_ids=protected_expert_ids,
            )
            for expert_id in expert_ids
        ]
        misses = [decision for decision in decisions if not decision.hit]

        self.hits += sum(decision.hit and not speculative for decision in decisions)
        self.useful_speculations += sum(
            decision.useful_speculation for decision in decisions
        )
        self.evictions += sum(
            decision.evicted_expert_id is not None for decision in misses
        )
        if speculative:
            self.speculative_loads += len(misses)
        else:
            self.demand_misses += len(misses)

        if not misses:
            return

        compute_stream = torch.cuda.current_stream(self.device)
        safe_to_overwrite = torch.cuda.Event(enable_timing=False)
        compute_stream.record_event(safe_to_overwrite)
        self.copy_stream.wait_event(safe_to_overwrite)

        with torch.cuda.stream(self.copy_stream):
            for decision in misses:
                if decision.evicted_expert_id is not None:
                    self.expert_to_slot[decision.evicted_expert_id] = -1
                self.gpu_w13_weight[decision.slot].copy_(
                    self.cpu_w13_weight[decision.expert_id], non_blocking=True
                )
                self.gpu_w2_weight[decision.slot].copy_(
                    self.cpu_w2_weight[decision.expert_id], non_blocking=True
                )
                self.slot_ready_events[decision.slot].record(self.copy_stream)
                self.bytes_copied += self.bytes_per_expert

        for decision in misses:
            self.expert_to_slot[decision.expert_id] = decision.slot

    @torch.compiler.disable
    def prefetch(self, predicted_expert_ids: torch.Tensor) -> None:
        """Asynchronously place predicted experts into GPU slots."""
        expert_ids = self._validate_ids(predicted_expert_ids.unique().tolist())
        self.prefetch_expert_ids(expert_ids)

    def prefetch_expert_ids(self, predicted_expert_ids: tuple[int, ...]) -> None:
        """Prefetch already-materialized IDs without another GPU synchronization."""
        expert_ids = self._validate_ids(predicted_expert_ids)
        admitted_ids, skipped = self.index.admit_speculative(expert_ids)
        self.skipped_speculations += skipped
        self._schedule(admitted_ids, speculative=True)

    def set_next_cache(self, next_cache: CachedExpertWeights | None) -> None:
        """Set the next MoE layer to receive same-ID predictions."""
        self.next_cache = next_cache

    @torch.compiler.disable
    def prepare(self, topk_ids: torch.Tensor) -> ExpertWeightResult:
        """Ensure exact router-selected experts are ready for kernel use."""
        expert_ids = self._validate_ids(topk_ids.unique().tolist())
        self._schedule(expert_ids, speculative=False)

        compute_stream = torch.cuda.current_stream(self.device)
        for expert_id in expert_ids:
            slot = self.index.entries[expert_id].slot
            compute_stream.wait_event(self.slot_ready_events[slot])

        remapped_ids = self.expert_to_slot[topk_ids.long()].to(topk_ids.dtype)
        if self.next_cache is not None:
            self.next_cache.prefetch_expert_ids(expert_ids)
        return ExpertWeightResult(
            w13_weight=self.gpu_w13_weight,
            w2_weight=self.gpu_w2_weight,
            topk_ids=remapped_ids,
        )

    def metrics(self) -> dict[str, int | float]:
        demand_accesses = self.hits + self.demand_misses
        hit_rate = self.hits / demand_accesses if demand_accesses > 0 else 0.0
        speculation_precision = (
            self.useful_speculations / self.speculative_loads
            if self.speculative_loads > 0
            else 0.0
        )
        return {
            "capacity": self.capacity,
            "num_experts": self.num_experts,
            "hits": self.hits,
            "demand_misses": self.demand_misses,
            "speculative_loads": self.speculative_loads,
            "skipped_speculations": self.skipped_speculations,
            "useful_speculations": self.useful_speculations,
            "evictions": self.evictions,
            "bytes_copied": self.bytes_copied,
            "cpu_bytes": self.cpu_bytes,
            "gpu_bytes": self.gpu_bytes,
            "hit_rate": hit_rate,
            "speculation_precision": speculation_precision,
        }

    def reset_metrics(self) -> None:
        """Reset counters without disturbing resident experts."""
        self.hits = 0
        self.demand_misses = 0
        self.speculative_loads = 0
        self.skipped_speculations = 0
        self.useful_speculations = 0
        self.evictions = 0
        self.bytes_copied = 0


def aggregate_expert_cache_metrics() -> dict[str, int | float]:
    """Aggregate metrics from all live expert caches in this process."""
    metrics = [cache.metrics() for cache in _ACTIVE_CACHES]
    hits = sum(int(item["hits"]) for item in metrics)
    demand_misses = sum(int(item["demand_misses"]) for item in metrics)
    speculative_loads = sum(int(item["speculative_loads"]) for item in metrics)
    skipped_speculations = sum(
        int(item["skipped_speculations"]) for item in metrics
    )
    useful_speculations = sum(int(item["useful_speculations"]) for item in metrics)
    demand_accesses = hits + demand_misses
    return {
        "layers": len(metrics),
        "capacity": sum(int(item["capacity"]) for item in metrics),
        "num_experts": sum(int(item["num_experts"]) for item in metrics),
        "hits": hits,
        "demand_misses": demand_misses,
        "speculative_loads": speculative_loads,
        "skipped_speculations": skipped_speculations,
        "useful_speculations": useful_speculations,
        "evictions": sum(int(item["evictions"]) for item in metrics),
        "bytes_copied": sum(int(item["bytes_copied"]) for item in metrics),
        "cpu_bytes": sum(int(item["cpu_bytes"]) for item in metrics),
        "gpu_bytes": sum(int(item["gpu_bytes"]) for item in metrics),
        "hit_rate": hits / demand_accesses if demand_accesses > 0 else 0.0,
        "speculation_precision": (
            useful_speculations / speculative_loads
            if speculative_loads > 0
            else 0.0
        ),
    }


def reset_expert_cache_metrics() -> None:
    """Reset metrics for every live expert cache in this process."""
    for cache in _ACTIVE_CACHES:
        cache.reset_metrics()
