# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm import SamplingParams
from vllm.v1.attention.backends.flex_attention import (
    FlexAttentionMetadata,
    causal_mask_mod,
    physical_to_logical_mapping,
)
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.cross_batch_attention import (
    CrossBatchAttentionData,
    CrossBatchAttentionMetadata,
    CrossBatchAttentionParams,
)
from vllm.v1.request import Request
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata


def make_params(
    group_id: str = "group-a",
    replica_id: int = 0,
    group_size: int = 2,
    virtual_token_id: int = 32000,
    virtual_window_size: int = 2,
) -> dict:
    return {
        "cross_batch_attention": {
            "enabled": True,
            "group_id": group_id,
            "replica_id": replica_id,
            "group_size": group_size,
            "virtual_token_id": virtual_token_id,
            "virtual_window_size": virtual_window_size,
        }
    }


def make_request(req_id: str, replica_id: int | None = None) -> Request:
    extra_args = make_params(replica_id=replica_id) if replica_id is not None else None
    return Request(
        request_id=req_id,
        prompt_token_ids=[1, 2, 32000],
        sampling_params=SamplingParams(max_tokens=1, extra_args=extra_args),
        pooling_params=None,
    )


def test_request_parses_cross_batch_extra_args():
    request = make_request("req-0", replica_id=0)

    assert request.cross_batch_attention == CrossBatchAttentionParams(
        group_id="group-a",
        replica_id=0,
        group_size=2,
        virtual_token_id=32000,
        virtual_window_size=2,
    )


def test_request_rejects_invalid_cross_batch_extra_args():
    with pytest.raises(ValueError, match="replica_id"):
        Request(
            request_id="bad",
            prompt_token_ids=[1],
            sampling_params=SamplingParams(
                max_tokens=1,
                extra_args=make_params(replica_id=2, group_size=2),
            ),
            pooling_params=None,
        )


def make_scheduler_for_group_test() -> Scheduler:
    scheduler = object.__new__(Scheduler)
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.skipped_waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.running = []
    scheduler.max_num_running_reqs = 4
    scheduler.max_model_len = 16
    scheduler.num_lookahead_tokens = 0
    scheduler.scheduler_config = SimpleNamespace(
        long_prefill_token_threshold=0,
        enable_chunked_prefill=True,
    )
    return scheduler


def test_scheduler_prepares_complete_cross_batch_group_at_front():
    scheduler = make_scheduler_for_group_test()
    req0 = make_request("req-0", replica_id=0)
    unrelated = make_request("unrelated")
    req1 = make_request("req-1", replica_id=1)
    scheduler.waiting.add_request(req0)
    scheduler.waiting.add_request(unrelated)
    scheduler.waiting.add_request(req1)

    assert scheduler._prepare_cross_batch_waiting_group(req0, token_budget=8)

    assert [req.request_id for req in scheduler.waiting][:3] == [
        "req-0",
        "req-1",
        "unrelated",
    ]


def test_scheduler_does_not_prepare_incomplete_cross_batch_group():
    scheduler = make_scheduler_for_group_test()
    req0 = make_request("req-0", replica_id=0)
    scheduler.waiting.add_request(req0)

    assert not scheduler._prepare_cross_batch_waiting_group(req0, token_budget=8)
    assert [req.request_id for req in scheduler.waiting] == ["req-0"]


class FakeBlockPool:

    def __init__(self, num_free_blocks: int):
        self.num_free_blocks = num_free_blocks

    def get_num_free_blocks(self) -> int:
        return self.num_free_blocks


class FakeKVCoordinator:

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: list,
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_tokens_main_model: int,
    ) -> int:
        return 1


class FakeKVCacheManager:

    def __init__(self, num_free_blocks: int):
        self.block_pool = FakeBlockPool(num_free_blocks)
        self.coordinator = FakeKVCoordinator()
        self.empty_kv_cache_blocks = SimpleNamespace(blocks=[])
        self.max_model_len = 16


def test_scheduler_does_not_prepare_cross_batch_group_without_kv_capacity():
    scheduler = make_scheduler_for_group_test()
    scheduler.kv_cache_manager = FakeKVCacheManager(num_free_blocks=1)
    req0 = make_request("req-0", replica_id=0)
    req1 = make_request("req-1", replica_id=1)
    scheduler.waiting.add_request(req0)
    scheduler.waiting.add_request(req1)

    assert not scheduler._prepare_cross_batch_waiting_group(req0, token_budget=8)
    assert [req.request_id for req in scheduler.waiting] == ["req-0", "req-1"]


def test_scheduler_prepares_cross_batch_group_when_kv_capacity_fits():
    scheduler = make_scheduler_for_group_test()
    scheduler.kv_cache_manager = FakeKVCacheManager(num_free_blocks=2)
    req0 = make_request("req-0", replica_id=0)
    req1 = make_request("req-1", replica_id=1)
    scheduler.waiting.add_request(req0)
    scheduler.waiting.add_request(req1)

    assert scheduler._prepare_cross_batch_waiting_group(req0, token_budget=8)
    assert [req.request_id for req in scheduler.waiting] == ["req-0", "req-1"]


def make_flex_metadata_for_mask(token_ids: torch.Tensor) -> FlexAttentionMetadata:
    metadata = object.__new__(FlexAttentionMetadata)
    metadata.doc_ids = torch.tensor([0], dtype=torch.int32)
    metadata.block_size = 2
    metadata.query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    metadata.decode_offset = torch.tensor([2, 2], dtype=torch.int32)
    metadata.seq_lens = torch.tensor([3, 3], dtype=torch.int32)
    metadata.physical_to_logical = torch.tensor(
        [
            [-1, -1],
            [-1, 1],
        ],
        dtype=torch.long,
    )
    metadata.logical_mask_mod = causal_mask_mod
    metadata.cross_batch_attention_metadata = CrossBatchAttentionMetadata(
        enabled=torch.tensor([True, True]),
        group_ids=torch.tensor([0, 0], dtype=torch.int32),
        replica_ids=torch.tensor([0, 1], dtype=torch.int32),
        allowed_peer_batches=torch.tensor([[1, -1], [0, -1]], dtype=torch.int32),
        allowed_peer_mask=torch.tensor([[True, False], [True, False]]),
        virtual_token_ids=torch.tensor([32000, 32000], dtype=torch.int32),
        virtual_window_sizes=torch.tensor([2, 2], dtype=torch.int32),
        token_ids=token_ids,
    )
    return metadata


def test_flex_cross_batch_mask_allows_peer_virtual_tokens():
    metadata = make_flex_metadata_for_mask(
        torch.tensor(
            [
                [1, 2, 32000],
                [3, 4, 32000],
            ],
            dtype=torch.int32,
        )
    )
    mask_mod = metadata.get_cross_batch_mask_mod()

    assert mask_mod(
        torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(2)
    )


def test_flex_cross_batch_mask_blocks_peer_non_virtual_tokens():
    metadata = make_flex_metadata_for_mask(
        torch.tensor(
            [
                [1, 2, 32000],
                [3, 4, 5],
            ],
            dtype=torch.int32,
        )
    )
    mask_mod = metadata.get_cross_batch_mask_mod()

    assert not mask_mod(
        torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(2)
    )


def test_flex_cross_batch_mask_handles_vectorized_queries():
    metadata = make_flex_metadata_for_mask(
        torch.tensor(
            [
                [1, 2, 32000],
                [3, 4, 32000],
            ],
            dtype=torch.int32,
        )
    )
    mask_mod = metadata.get_cross_batch_mask_mod()

    result = mask_mod(
        torch.tensor(0),
        torch.tensor(0),
        torch.tensor([0, 0]),
        torch.tensor([0, 2]),
    )

    assert result.tolist() == [False, True]


def test_flex_cross_batch_mask_uses_absolute_virtual_positions():
    metadata = make_flex_metadata_for_mask(
        torch.tensor(
            [
                [1, 2, 32000, 3, 32000],
                [4, 5, 32000, 6, 32000],
            ],
            dtype=torch.int32,
        )
    )
    metadata.block_size = 1
    metadata.query_start_loc = torch.tensor([0, 5], dtype=torch.int32)
    metadata.decode_offset = torch.tensor([2, 0], dtype=torch.int32)
    metadata.seq_lens = torch.tensor([5, 5], dtype=torch.int32)
    metadata.physical_to_logical = torch.tensor(
        [
            [-1, -1, -1, -1, -1, -1],
            [-1, 0, 1, 2, 3, 4],
        ],
        dtype=torch.long,
    )
    metadata.cross_batch_attention_metadata.virtual_window_sizes = torch.tensor(
        [2, 2], dtype=torch.int32
    )
    mask_mod = metadata.get_cross_batch_mask_mod()

    # Query request 0 is at absolute logical position 2. Peer request 1 has
    # virtual tokens at positions 2 and 4. The later peer virtual token must be
    # blocked even though 2 % 2 == 4 % 2.
    assert mask_mod(
        torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(3)
    )
    assert not mask_mod(
        torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(5)
    )


def test_flex_cross_batch_mask_can_use_position_scheduled_virtual_tokens():
    metadata = make_flex_metadata_for_mask(
        torch.tensor(
            [
                [10, 11, 12, 13, 14],
                [20, 21, 22, 23, 24],
            ],
            dtype=torch.int32,
        )
    )
    metadata.block_size = 1
    metadata.query_start_loc = torch.tensor([0, 5], dtype=torch.int32)
    metadata.decode_offset = torch.tensor([2, 0], dtype=torch.int32)
    metadata.seq_lens = torch.tensor([5, 5], dtype=torch.int32)
    metadata.physical_to_logical = torch.tensor(
        [
            [-1, -1, -1, -1, -1, -1],
            [-1, 0, 1, 2, 3, 4],
        ],
        dtype=torch.long,
    )
    metadata.cross_batch_attention_metadata.virtual_token_ids = torch.tensor(
        [-1, -1], dtype=torch.int32
    )
    metadata.cross_batch_attention_metadata.virtual_window_sizes = torch.tensor(
        [2, 2], dtype=torch.int32
    )
    mask_mod = metadata.get_cross_batch_mask_mod()

    assert mask_mod(
        torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(3)
    )
    assert not mask_mod(
        torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(4)
    )


def test_gpu_model_runner_builds_cross_batch_metadata_in_batch_order():
    runner = object.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.input_batch = SimpleNamespace(
        req_ids=["req-1", "req-0"],
        token_ids_cpu_tensor=torch.tensor(
            [
                [10, 99, 11, 0],
                [20, 99, 21, 0],
                [0, 0, 0, 0],
            ],
            dtype=torch.int32,
        ),
    )
    scheduler_output = SimpleNamespace(
        cross_batch_attention_data=CrossBatchAttentionData(
            params_by_req_id={
                "req-0": CrossBatchAttentionParams(
                    group_id="group-a",
                    replica_id=0,
                    group_size=2,
                    virtual_token_id=99,
                    virtual_window_size=2,
                ),
                "req-1": CrossBatchAttentionParams(
                    group_id="group-a",
                    replica_id=1,
                    group_size=2,
                    virtual_token_id=99,
                    virtual_window_size=2,
                ),
            }
        )
    )

    metadata = runner._make_cross_batch_attention_metadata(
        scheduler_output=scheduler_output,
        num_reqs=2,
        num_reqs_padded=3,
    )

    assert metadata is not None
    assert metadata.enabled.tolist() == [True, True, False]
    assert metadata.replica_ids.tolist() == [1, 0, -1]
    assert metadata.allowed_peer_batches.tolist() == [
        [1, -1, -1],
        [0, -1, -1],
        [-1, -1, -1],
    ]
    assert metadata.allowed_peer_mask.tolist() == [
        [True, False, False],
        [True, False, False],
        [False, False, False],
    ]
    assert metadata.token_ids.tolist() == [
        [10, 99, 11, 0],
        [20, 99, 21, 0],
        [0, 0, 0, 0],
    ]


def make_flex_metadata_for_dense_reference(
    cross_batch_attention_metadata: CrossBatchAttentionMetadata | None,
) -> FlexAttentionMetadata:
    block_size = 1
    block_table = torch.tensor(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ],
        dtype=torch.int32,
    )
    seq_lens = torch.tensor([4, 4], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 4, 8], dtype=torch.int32)
    metadata = FlexAttentionMetadata(
        causal=True,
        num_actual_tokens=8,
        max_query_len=4,
        query_start_loc=query_start_loc,
        max_seq_len=4,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=torch.arange(8, dtype=torch.int64),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        total_cache_tokens=9,
        block_size=block_size,
        max_possible_sequence_length=4,
        num_reqs=2,
        physical_to_logical=physical_to_logical_mapping(
            block_table=block_table,
            seq_lens=seq_lens,
            block_size=block_size,
            total_blocks=9,
        ),
        decode_offset=torch.tensor([0, 0], dtype=torch.int32),
        num_blocks_per_seq=torch.tensor([4, 4], dtype=torch.int32),
        direct_build=False,
        cross_batch_attention_metadata=cross_batch_attention_metadata,
    )
    return metadata


def dense_attention_reference(
    metadata: FlexAttentionMetadata,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    mask_mod = metadata.get_mask_mod()
    kv_idx = torch.arange(metadata.total_cache_tokens)[None, :]
    mask = torch.cat(
        [
            mask_mod(torch.tensor(0), torch.tensor(0), torch.tensor([[q]]), kv_idx)
            for q in range(metadata.num_actual_tokens)
        ],
        dim=0,
    )
    scores = query @ key.T
    scores = scores.masked_fill(~mask, float("-inf"))
    return F.softmax(scores, dim=-1) @ value


def test_cross_batch_attention_changes_only_virtual_query_outputs():
    token_ids = torch.tensor(
        [
            [10, 99, 11, 99],
            [20, 99, 21, 99],
        ],
        dtype=torch.int32,
    )
    cross_batch = CrossBatchAttentionMetadata(
        enabled=torch.tensor([True, True]),
        group_ids=torch.tensor([0, 0], dtype=torch.int32),
        replica_ids=torch.tensor([0, 1], dtype=torch.int32),
        allowed_peer_batches=torch.tensor([[1, -1], [0, -1]], dtype=torch.int32),
        allowed_peer_mask=torch.tensor([[True, False], [True, False]]),
        virtual_token_ids=torch.tensor([99, 99], dtype=torch.int32),
        virtual_window_sizes=torch.tensor([4, 4], dtype=torch.int32),
        token_ids=token_ids,
    )
    baseline_metadata = make_flex_metadata_for_dense_reference(None)
    cross_metadata = make_flex_metadata_for_dense_reference(cross_batch)

    query = torch.eye(8, dtype=torch.float32)
    key = torch.zeros(9, 8, dtype=torch.float32)
    value = torch.zeros(9, 3, dtype=torch.float32)
    for logical_idx, physical_idx in enumerate([1, 2, 3, 4]):
        key[physical_idx, logical_idx] = 1.0
        value[physical_idx] = torch.tensor([float(logical_idx), 0.0, 0.0])
    for logical_idx, physical_idx in enumerate([5, 6, 7, 8], start=4):
        key[physical_idx, logical_idx] = 1.0
        value[physical_idx] = torch.tensor([float(logical_idx), 10.0, 0.0])

    baseline = dense_attention_reference(baseline_metadata, query, key, value)
    enabled = dense_attention_reference(cross_metadata, query, key, value)

    non_virtual_query_indices = torch.tensor([0, 2, 4, 6])
    virtual_query_indices = torch.tensor([1, 3, 5, 7])

    torch.testing.assert_close(
        enabled[non_virtual_query_indices],
        baseline[non_virtual_query_indices],
    )
    assert not torch.allclose(
        enabled[virtual_query_indices],
        baseline[virtual_query_indices],
    )


class FakeNonFlexBackend:

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"


def test_cross_batch_attention_rejects_non_flex_backend():
    cross_batch = CrossBatchAttentionMetadata(
        enabled=torch.tensor([True]),
        group_ids=torch.tensor([0], dtype=torch.int32),
        replica_ids=torch.tensor([0], dtype=torch.int32),
        allowed_peer_batches=torch.tensor([[-1]], dtype=torch.int32),
        allowed_peer_mask=torch.tensor([[False]]),
        virtual_token_ids=torch.tensor([99], dtype=torch.int32),
        virtual_window_sizes=torch.tensor([4], dtype=torch.int32),
        token_ids=torch.tensor([[99]], dtype=torch.int32),
    )

    with pytest.raises(ValueError, match="FLEX_ATTENTION"):
        build_attn_metadata(
            attn_groups=[
                [
                    SimpleNamespace(
                        backend=FakeNonFlexBackend,
                        get_metadata_builder=lambda _: None,
                        layer_names=["layer"],
                    )
                ]
            ],
            num_reqs=1,
            num_tokens=1,
            query_start_loc_gpu=torch.tensor([0, 1], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
            max_query_len=1,
            seq_lens=torch.tensor([1], dtype=torch.int32),
            max_seq_len=1,
            block_tables=[torch.tensor([[1]], dtype=torch.int32)],
            slot_mappings=torch.tensor([[0]], dtype=torch.int64),
            kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
            cross_batch_attention_metadata=cross_batch,
        )
