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
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import torch

_ACTIVE_CACHES: weakref.WeakSet[CachedExpertWeights] = weakref.WeakSet()
_STAGING_TASKS: dict[
    str,
    tuple[
        ThreadPoolExecutor,
        Future[tuple[tuple[int, ...], ...]],
    ],
] = {}


@dataclass(frozen=True)
class ExecutionPoint:
    """Logical deadline or retention boundary for one agent session."""

    token_index: int
    layer_index: int


@dataclass(frozen=True)
class ExpertResidencyIntent:
    """Layer-local expert residency request emitted by any policy."""

    layer_index: int
    expert_ids: tuple[int, ...]
    ready_by: ExecutionPoint
    retain_until: ExecutionPoint | None
    priority: float
    group_id: str


@dataclass(frozen=True)
class ResidencyTicket:
    """Identifier for a staged single-session residency group."""

    group_id: str
    layer_indices: tuple[int, ...]
    routing_mode: Literal["fallback", "restricted"]


@dataclass
class OracleLookaheadState:
    """Recorded decode routes used only for controlled lookahead benchmarks."""

    routes: tuple[tuple[tuple[int, ...], ...], ...]
    caches: tuple[CachedExpertWeights, ...]
    lookahead_layers: int
    calls_by_layer: list[int]

    def prefetch_from(self, layer_index: int) -> None:
        token_index = self.calls_by_layer[layer_index]
        self.calls_by_layer[layer_index] += 1
        if self.lookahead_layers == 0:
            return
        target_position = (
            token_index * len(self.caches)
            + layer_index
            + self.lookahead_layers
        )
        target_token = target_position // len(self.caches)
        target_layer = target_position % len(self.caches)
        if target_token >= len(self.routes):
            return
        self.caches[target_layer].prefetch_replacing_expert_ids(
            self.routes[target_token][target_layer]
        )


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
        self.retained_expert_ids: frozenset[int] = frozenset()

    def _victim(self, protected_expert_ids: frozenset[int]) -> int:
        if not self.entries:
            raise RuntimeError("cannot evict from an empty cache")

        candidates = self.entries.keys() - protected_expert_ids
        if not candidates:
            raise RuntimeError("no evictable cache slot remains")
        victim_id = min(
            candidates,
            key=lambda expert_id: (
                expert_id in self.retained_expert_ids,
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
            self.retained_expert_ids = (
                self.retained_expert_ids - {evicted_expert_id}
            )

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

    def set_retained_experts(self, expert_ids: Iterable[int]) -> None:
        """Prefer a resident plan while allowing correctness demand to win."""
        retained = frozenset(expert_ids)
        missing = retained - self.entries.keys()
        if missing:
            raise RuntimeError(
                f"cannot retain nonresident experts: {tuple(sorted(missing))}"
            )
        self.retained_expert_ids = retained

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
        experts_per_token: int = 1,
    ) -> None:
        if w13_weight.shape[0] != w2_weight.shape[0]:
            raise ValueError(
                "w13_weight and w2_weight must have the same expert dimension"
            )
        if w13_weight.device.type != "cuda" or w2_weight.device.type != "cuda":
            raise ValueError("initial expert weights must be CUDA tensors")

        self.num_experts = int(w13_weight.shape[0])
        self.capacity = min(capacity, self.num_experts)
        if not 0 < experts_per_token <= self.num_experts:
            raise ValueError(
                f"experts_per_token must be in [1, {self.num_experts}], got "
                f"{experts_per_token}"
            )
        self.experts_per_token = experts_per_token
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
        self.plan_loads = 0
        self.layer_index: int | None = None
        self.active_group_id: str | None = None
        self.staged_group_id: str | None = None
        self.staged_expert_ids: tuple[int, ...] = ()
        self.routing_mode: Literal["fallback", "restricted"] = "fallback"
        self.allowed_expert_mask = torch.ones(
            (self.num_experts,), dtype=torch.bool, device=self.device
        )
        self.next_cache: CachedExpertWeights | None = None
        self.oracle_lookahead_state: OracleLookaheadState | None = None
        self.route_recording_enabled = False
        self.recorded_routes: list[tuple[int, ...]] = []
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

    def _schedule(
        self,
        expert_ids: tuple[int, ...],
        *,
        speculative: bool,
    ) -> tuple[int, ...]:
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
        demand_wait_ids = tuple(
            decision.expert_id
            for decision in decisions
            if not speculative and (not decision.hit or decision.useful_speculation)
        )

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
            return demand_wait_ids

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
        return demand_wait_ids

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

    def prefetch_replacing_expert_ids(
        self,
        predicted_expert_ids: tuple[int, ...],
    ) -> None:
        """Prefetch an oracle set, allowing it to replace dynamic entries."""
        expert_ids = self._validate_ids(predicted_expert_ids)
        self._schedule(expert_ids, speculative=True)

    def set_next_cache(self, next_cache: CachedExpertWeights | None) -> None:
        """Set the next MoE layer to receive same-ID predictions."""
        self.next_cache = next_cache

    def set_layer_index(self, layer_index: int) -> None:
        """Assign deterministic model order after every cache is constructed."""
        if layer_index < 0:
            raise ValueError("layer_index must be non-negative")
        self.layer_index = layer_index

    def stage_residency_group(
        self,
        group_id: str,
        expert_ids: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Load a complete layer plan into the one-bank cache.

        The caller must ensure model execution is idle until activation. This
        first implementation may evict the previous plan while staging.
        """
        validated_ids = self._validate_ids(expert_ids)
        if self.staged_group_id is not None:
            raise RuntimeError(
                f"group {self.staged_group_id!r} is already staged; activate or "
                "cancel it before staging another group"
            )
        self.active_group_id = None
        self.routing_mode = "fallback"
        self.allowed_expert_mask.fill_(True)
        self.index.set_retained_experts(())
        copied_before = self.bytes_copied
        self._schedule(validated_ids, speculative=True)
        if self.bytes_copied > copied_before:
            self.plan_loads += 1
        self.staged_group_id = group_id
        self.staged_expert_ids = validated_ids
        return validated_ids

    def wait_for_experts(self, expert_ids: tuple[int, ...]) -> None:
        """Make the current compute stream wait for staged expert copies."""
        compute_stream = torch.cuda.current_stream(self.device)
        for expert_id in expert_ids:
            slot = self.index.entries[expert_id].slot
            compute_stream.wait_event(self.slot_ready_events[slot])

    def activate_residency_group(
        self,
        group_id: str,
        expert_ids: tuple[int, ...],
        routing_mode: Literal["fallback", "restricted"],
    ) -> None:
        """Publish a fully staged plan and its optional router restriction."""
        if self.staged_group_id != group_id:
            raise RuntimeError(
                f"staged group {self.staged_group_id!r} does not match {group_id!r}"
            )
        if expert_ids != self.staged_expert_ids:
            raise RuntimeError("activated expert IDs do not match the staged plan")
        if routing_mode == "restricted" and len(expert_ids) < self.experts_per_token:
            raise ValueError(
                f"restricted routing needs at least {self.experts_per_token} experts, "
                f"got {len(expert_ids)}"
            )
        self.index.set_retained_experts(expert_ids)
        for expert_id in expert_ids:
            self.index.entries[expert_id].speculative = False
        self.allowed_expert_mask.fill_(routing_mode == "fallback")
        if routing_mode == "restricted":
            indices = torch.tensor(expert_ids, dtype=torch.long, device=self.device)
            self.allowed_expert_mask[indices] = True
        self.routing_mode = routing_mode
        self.active_group_id = group_id
        self.staged_group_id = None
        self.staged_expert_ids = ()

    def cancel_residency_group(self, group_id: str) -> None:
        """Release soft retention and restore unrestricted routing."""
        if self.staged_group_id == group_id:
            self.staged_group_id = None
            self.staged_expert_ids = ()
        if self.active_group_id != group_id:
            return
        self.index.set_retained_experts(())
        self.allowed_expert_mask.fill_(True)
        self.active_group_id = None
        self.routing_mode = "fallback"

    def reset_contents(self) -> None:
        """Clear residency state while preserving allocated CPU/GPU buffers."""
        self.index = LFRUCacheIndex(self.capacity)
        self.expert_to_slot.fill_(-1)
        self.active_group_id = None
        self.staged_group_id = None
        self.staged_expert_ids = ()
        self.routing_mode = "fallback"
        self.allowed_expert_mask.fill_(True)
        self.oracle_lookahead_state = None
        self.route_recording_enabled = False
        self.recorded_routes = []
        self.reset_metrics()

    def apply_router_mask(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Restrict routing to an activated plan without changing tensor shape."""
        if self.routing_mode == "fallback":
            return router_logits
        if router_logits.shape[-1] != self.num_experts:
            raise RuntimeError(
                f"router has {router_logits.shape[-1]} experts, cache has "
                f"{self.num_experts}"
            )
        return router_logits.masked_fill(~self.allowed_expert_mask, float("-inf"))

    @torch.compiler.disable
    def prepare(self, topk_ids: torch.Tensor) -> ExpertWeightResult:
        """Ensure exact router-selected experts are ready for kernel use."""
        if self.route_recording_enabled:
            if topk_ids.shape[0] != 1:
                raise RuntimeError(
                    "route recording supports decode batch size one"
                )
            self.recorded_routes.append(
                tuple(int(expert_id) for expert_id in topk_ids.reshape(-1).tolist())
            )
        if self.oracle_lookahead_state is not None:
            if self.layer_index is None:
                raise RuntimeError("oracle lookahead requires a linked layer index")
            if topk_ids.shape[0] != 1:
                raise RuntimeError(
                    "oracle lookahead benchmark supports decode batch size one"
                )
            self.oracle_lookahead_state.prefetch_from(self.layer_index)
        plan_covers_routes = self.active_group_id is not None and (
            self.routing_mode == "restricted"
            or len(self.index.retained_expert_ids) == self.num_experts
        )
        if plan_covers_routes and self.next_cache is None:
            self.hits += topk_ids.numel()
            return ExpertWeightResult(
                w13_weight=self.gpu_w13_weight,
                w2_weight=self.gpu_w2_weight,
                topk_ids=self.expert_to_slot[topk_ids.long()].to(topk_ids.dtype),
            )

        logical_ids = topk_ids.reshape(-1).tolist()
        expert_ids = self._validate_ids(tuple(dict.fromkeys(logical_ids)))
        missed_expert_ids = self._schedule(expert_ids, speculative=False)

        self.wait_for_experts(missed_expert_ids)

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
            "plan_loads": self.plan_loads,
            "retained_experts": len(self.index.retained_expert_ids),
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
        self.plan_loads = 0


def _ordered_active_caches() -> tuple[CachedExpertWeights, ...]:
    caches = tuple(_ACTIVE_CACHES)
    if not caches:
        raise RuntimeError("no live expert caches are registered")
    if any(cache.layer_index is None for cache in caches):
        raise RuntimeError("expert cache layer indices have not been linked")
    ordered = tuple(sorted(caches, key=lambda cache: int(cache.layer_index)))
    indices = tuple(int(cache.layer_index) for cache in ordered)
    if indices != tuple(range(len(ordered))):
        raise RuntimeError(f"expert cache layer indices are not contiguous: {indices}")
    return ordered


def reset_expert_weight_cache_contents() -> None:
    """Synchronously clear all slots for a controlled single-worker replay."""
    if _STAGING_TASKS:
        raise RuntimeError("cannot reset while an expert group is staging")
    caches = _ordered_active_caches()
    torch.cuda.synchronize(caches[0].device)
    for cache in caches:
        cache.reset_contents()


def enable_expert_route_recording() -> None:
    """Record exact per-token, per-layer decode routes for oracle replay."""
    caches = _ordered_active_caches()
    if any(cache.route_recording_enabled for cache in caches):
        raise RuntimeError("expert route recording is already enabled")
    for cache in caches:
        cache.recorded_routes = []
        cache.route_recording_enabled = True


def disable_expert_route_recording(
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Stop route recording and return routes indexed by token then layer."""
    caches = _ordered_active_caches()
    if not all(cache.route_recording_enabled for cache in caches):
        raise RuntimeError("expert route recording is not enabled on every layer")
    calls_by_layer = tuple(len(cache.recorded_routes) for cache in caches)
    if len(set(calls_by_layer)) != 1:
        raise RuntimeError(
            f"route recording calls differ by layer: {calls_by_layer}"
        )
    for cache in caches:
        cache.route_recording_enabled = False
    return tuple(
        tuple(cache.recorded_routes[token_index] for cache in caches)
        for token_index in range(calls_by_layer[0])
    )


def configure_oracle_expert_lookahead(
    routes: tuple[tuple[tuple[int, ...], ...], ...],
    lookahead_layers: int,
) -> None:
    """Enable recorded-route lookahead instrumentation for decode benchmarks."""
    if lookahead_layers < 0:
        raise ValueError("lookahead_layers must be non-negative")
    caches = _ordered_active_caches()
    if not routes:
        raise ValueError("at least one token of recorded routes is required")
    if any(len(token_routes) != len(caches) for token_routes in routes):
        raise ValueError("every recorded token must specify every MoE layer")
    for token_routes in routes:
        for layer_index, expert_ids in enumerate(token_routes):
            caches[layer_index]._validate_ids(expert_ids)
    state = OracleLookaheadState(
        routes=routes,
        caches=caches,
        lookahead_layers=lookahead_layers,
        calls_by_layer=[0] * len(caches),
    )
    for cache in caches:
        cache.oracle_lookahead_state = state


def disable_oracle_expert_lookahead() -> tuple[int, ...]:
    """Disable instrumentation and return decode calls observed per layer."""
    caches = _ordered_active_caches()
    state = caches[0].oracle_lookahead_state
    if state is None:
        raise RuntimeError("oracle lookahead is not enabled")
    calls = tuple(state.calls_by_layer)
    for cache in caches:
        cache.oracle_lookahead_state = None
    return calls


def _validate_residency_intents(
    intents: tuple[ExpertResidencyIntent, ...],
) -> tuple[
    str,
    tuple[CachedExpertWeights, ...],
    dict[int, ExpertResidencyIntent],
]:
    if not intents:
        raise ValueError("at least one residency intent is required")
    group_ids = {intent.group_id for intent in intents}
    if len(group_ids) != 1:
        raise ValueError("all residency intents must share one group_id")
    group_id = next(iter(group_ids))
    caches = _ordered_active_caches()
    by_layer = {intent.layer_index: intent for intent in intents}
    if len(by_layer) != len(intents):
        raise ValueError("a replacement group contains duplicate layer intents")
    expected_layers = set(range(len(caches)))
    if set(by_layer) != expected_layers:
        raise ValueError(
            "a replacement group must specify every MoE layer; expected "
            f"{tuple(sorted(expected_layers))}, got {tuple(sorted(by_layer))}"
        )
    for layer_index in range(len(caches)):
        intent = by_layer[layer_index]
        if intent.ready_by.layer_index != layer_index:
            raise ValueError(
                f"layer {layer_index} intent has deadline for layer "
                f"{intent.ready_by.layer_index}"
            )
        caches[layer_index]._validate_ids(intent.expert_ids)
        if caches[layer_index].staged_group_id is not None:
            raise RuntimeError(
                f"layer {layer_index} already has staged group "
                f"{caches[layer_index].staged_group_id!r}"
            )
    return group_id, caches, by_layer


def stage_expert_residency_group(
    intents: tuple[ExpertResidencyIntent, ...],
    *,
    routing_mode: Literal["fallback", "restricted"] = "fallback",
) -> ResidencyTicket:
    """Start staging one model-wide plan on a background worker thread.

    This one-bank implementation can overwrite experts from the active plan.
    The engine therefore must remain idle from staging through activation.
    """
    group_id, caches, by_layer = _validate_residency_intents(intents)
    if _STAGING_TASKS:
        raise RuntimeError(
            "one residency group is already staged; activate or cancel it first"
        )

    def stage_all_layers() -> tuple[tuple[int, ...], ...]:
        return tuple(
            cache.stage_residency_group(
                group_id,
                by_layer[layer_index].expert_ids,
            )
            for layer_index, cache in enumerate(caches)
        )

    executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="vllm-expert-stage",
    )
    future = executor.submit(stage_all_layers)
    _STAGING_TASKS[group_id] = (executor, future)
    return ResidencyTicket(
        group_id=group_id,
        layer_indices=tuple(range(len(caches))),
        routing_mode=routing_mode,
    )


def activate_expert_residency_group(ticket: ResidencyTicket) -> None:
    """Wait for a staged model-wide plan and make it authoritative."""
    caches = _ordered_active_caches()
    expected_layers = tuple(range(len(caches)))
    if ticket.layer_indices != expected_layers:
        raise ValueError(
            f"ticket layers {ticket.layer_indices} do not match {expected_layers}"
        )
    task = _STAGING_TASKS.pop(ticket.group_id, None)
    if task is None:
        raise RuntimeError(f"group {ticket.group_id!r} has not been staged")
    executor, future = task
    staged_ids = future.result()
    executor.shutdown(wait=True)
    for cache, expert_ids in zip(caches, staged_ids):
        if cache.staged_group_id != ticket.group_id:
            raise RuntimeError(
                f"layer {cache.layer_index} staged group "
                f"{cache.staged_group_id!r} does not match {ticket.group_id!r}"
            )
        cache.wait_for_experts(expert_ids)
    torch.cuda.current_stream(caches[0].device).synchronize()
    for cache, expert_ids in zip(caches, staged_ids):
        cache.activate_residency_group(
            ticket.group_id,
            expert_ids,
            ticket.routing_mode,
        )


def replace_expert_residency_group(
    intents: tuple[ExpertResidencyIntent, ...],
    *,
    routing_mode: Literal["fallback", "restricted"] = "fallback",
) -> ResidencyTicket:
    """Blocking one-bank replacement for a single active agent session."""
    ticket = stage_expert_residency_group(
        intents,
        routing_mode=routing_mode,
    )
    activate_expert_residency_group(ticket)
    return ticket


def cancel_expert_residency_group(group_id: str) -> None:
    """Cancel the active single-session group on every cached MoE layer."""
    task = _STAGING_TASKS.pop(group_id, None)
    if task is not None:
        executor, future = task
        future.result()
        executor.shutdown(wait=True)
    for cache in _ordered_active_caches():
        cache.cancel_residency_group(group_id)


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
        "plan_loads": sum(int(item["plan_loads"]) for item in metrics),
        "retained_experts": sum(int(item["retained_experts"]) for item in metrics),
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
