# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm import SamplingParams
from vllm.v1.attention.backends.flex_attention import (
    FlexAttentionMetadata,
    causal_mask_mod,
)
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.cross_batch_attention import (
    CrossBatchAttentionMetadata,
    CrossBatchAttentionParams,
)
from vllm.v1.request import Request


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
