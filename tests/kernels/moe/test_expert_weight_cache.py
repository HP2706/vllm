# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch

from vllm.model_executor.layers.fused_moe.expert_weight_cache import (
    CachedExpertWeights,
    LFRUCacheIndex,
    disable_expert_route_recording,
    enable_expert_route_recording,
)


def test_lfru_cache_index_reuses_slots_and_tracks_hits() -> None:
    index = LFRUCacheIndex(capacity=2)

    first = index.resolve(4, speculative=False)
    second = index.resolve(7, speculative=False)
    repeated = index.resolve(4, speculative=False)

    assert not first.hit
    assert not second.hit
    assert repeated.hit
    assert repeated.slot == first.slot
    assert index.resident_experts() == (4, 7)


def test_lfru_cache_index_recognizes_useful_speculation() -> None:
    index = LFRUCacheIndex(capacity=2)

    prefetched = index.resolve(3, speculative=True)
    demanded = index.resolve(3, speculative=False)
    demanded_again = index.resolve(3, speculative=False)

    assert not prefetched.hit
    assert demanded.hit
    assert demanded.useful_speculation
    assert demanded_again.hit
    assert not demanded_again.useful_speculation


def test_lfru_cache_index_evicts_low_frequency_entry() -> None:
    index = LFRUCacheIndex(capacity=2)
    index.resolve(0, speculative=False)
    index.resolve(1, speculative=False)
    for _ in range(5):
        index.resolve(0, speculative=False)

    replacement = index.resolve(2, speculative=False)

    assert replacement.evicted_expert_id == 1
    assert index.resident_experts() == (0, 2)


def test_lfru_cache_index_does_not_evict_a_requested_hit() -> None:
    index = LFRUCacheIndex(capacity=2)
    index.resolve(0, speculative=False)
    for _ in range(5):
        index.resolve(1, speculative=False)

    requested = frozenset({0, 2})
    hit = index.resolve(
        0, speculative=False, protected_expert_ids=requested
    )
    miss = index.resolve(
        2, speculative=False, protected_expert_ids=requested
    )

    assert hit.hit
    assert miss.evicted_expert_id == 1
    assert index.resident_experts() == (0, 2)


def test_lfru_cache_index_admits_speculation_only_into_free_slots() -> None:
    index = LFRUCacheIndex(capacity=3)
    index.resolve(0, speculative=False)
    index.resolve(1, speculative=False)

    admitted, skipped = index.admit_speculative((0, 2, 3, 4))

    assert admitted == (2,)
    assert skipped == 2
    assert index.resident_experts() == (0, 1)


def test_lfru_cache_index_prefers_to_keep_retained_experts() -> None:
    index = LFRUCacheIndex(capacity=2)
    index.resolve(0, speculative=True)
    index.resolve(1, speculative=False)
    index.set_retained_experts((0,))

    replacement = index.resolve(2, speculative=False)

    assert replacement.evicted_expert_id == 1
    assert index.resident_experts() == (0, 2)
    assert index.retained_expert_ids == frozenset({0})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cached_expert_weights_cold_miss_and_warm_hit() -> None:
    torch.manual_seed(2706)
    w13 = torch.randn(8, 32, 16, dtype=torch.bfloat16, device="cuda")
    w2 = torch.randn(8, 16, 16, dtype=torch.bfloat16, device="cuda")
    expected_w13 = w13.cpu()
    expected_w2 = w2.cpu()
    cache = CachedExpertWeights(capacity=4, w13_weight=w13, w2_weight=w2)
    ids = torch.tensor([[1, 3, 5, 7]], dtype=torch.int32, device="cuda")

    cold = cache.prepare(ids)
    torch.cuda.synchronize()
    for expert_id, slot in zip(ids[0].tolist(), cold.topk_ids[0].tolist()):
        torch.testing.assert_close(cold.w13_weight[slot].cpu(), expected_w13[expert_id])
        torch.testing.assert_close(cold.w2_weight[slot].cpu(), expected_w2[expert_id])

    warm = cache.prepare(ids)
    torch.cuda.synchronize()
    assert torch.equal(cold.topk_ids, warm.topk_ids)
    assert cache.metrics()["demand_misses"] == 4
    assert cache.metrics()["hits"] == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cached_expert_weights_oracle_prefetch_avoids_demand_miss() -> None:
    w13 = torch.randn(8, 32, 16, dtype=torch.bfloat16, device="cuda")
    w2 = torch.randn(8, 16, 16, dtype=torch.bfloat16, device="cuda")
    cache = CachedExpertWeights(capacity=4, w13_weight=w13, w2_weight=w2)
    ids = torch.tensor([[0, 2, 4, 6]], dtype=torch.int32, device="cuda")

    cache.prefetch(ids)
    result = cache.prepare(ids)
    torch.cuda.synchronize()

    assert result.topk_ids.min().item() >= 0
    assert result.topk_ids.max().item() < cache.capacity
    metrics = cache.metrics()
    assert metrics["speculative_loads"] == 4
    assert metrics["useful_speculations"] == 4
    assert metrics["demand_misses"] == 0
    assert metrics["speculation_precision"] == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cached_expert_weights_records_exact_decode_routes() -> None:
    w13 = torch.randn(8, 32, 16, dtype=torch.bfloat16, device="cuda")
    w2 = torch.randn(8, 16, 16, dtype=torch.bfloat16, device="cuda")
    cache = CachedExpertWeights(capacity=4, w13_weight=w13, w2_weight=w2)
    cache.set_layer_index(0)
    first_ids = torch.tensor([[6, 1, 4, 2]], dtype=torch.int32, device="cuda")
    second_ids = torch.tensor([[3, 7, 0, 5]], dtype=torch.int32, device="cuda")

    enable_expert_route_recording()
    cache.prepare(first_ids)
    cache.prepare(second_ids)
    routes = disable_expert_route_recording()
    torch.cuda.synchronize()

    assert routes == (((6, 1, 4, 2),), ((3, 7, 0, 5),))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cached_expert_weights_activates_restricted_residency_group() -> None:
    w13 = torch.randn(8, 32, 16, dtype=torch.bfloat16, device="cuda")
    w2 = torch.randn(8, 16, 16, dtype=torch.bfloat16, device="cuda")
    cache = CachedExpertWeights(
        capacity=4,
        w13_weight=w13,
        w2_weight=w2,
        experts_per_token=2,
    )

    expert_ids = cache.stage_residency_group("turn-1", (1, 3, 5, 7))
    assert cache.active_group_id is None
    assert cache.staged_group_id == "turn-1"
    cache.wait_for_experts(expert_ids)
    torch.cuda.synchronize()
    cache.activate_residency_group("turn-1", expert_ids, "restricted")
    assert cache.active_group_id == "turn-1"
    assert cache.staged_group_id is None
    logits = torch.zeros((2, 8), dtype=torch.float32, device="cuda")
    masked = cache.apply_router_mask(logits)

    assert torch.isfinite(masked[:, [1, 3, 5, 7]]).all()
    assert torch.isneginf(masked[:, [0, 2, 4, 6]]).all()
    assert cache.index.retained_expert_ids == frozenset({1, 3, 5, 7})
    assert all(
        not cache.index.entries[expert_id].speculative
        for expert_id in expert_ids
    )

    routed_ids = torch.tensor([[1, 5]], dtype=torch.int32, device="cuda")
    result = cache.prepare(routed_ids)
    torch.cuda.synchronize()
    assert torch.equal(
        result.topk_ids,
        cache.expert_to_slot[routed_ids.long()].to(routed_ids.dtype),
    )
    assert cache.metrics()["hits"] == routed_ids.numel()

    cache.cancel_residency_group("turn-1")
    assert torch.equal(cache.apply_router_mask(logits), logits)
