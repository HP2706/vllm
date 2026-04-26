# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for FlexAttention backend vs default backend"""

import pytest
import torch
from packaging import version

from tests.utils import set_random_seed
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
)
from vllm.v1.attention.backends.flex_attention import (
    FlexAttentionMetadata,
    FlexAttentionMetadataBuilder,
    physical_to_logical_mapping,
)
from vllm.v1.cross_batch_attention import CrossBatchAttentionMetadata

from ..models.utils import check_embeddings_close, check_logprobs_close

TORCH_VERSION = version.parse(torch.__version__)
MINIMUM_TORCH_VERSION = version.parse("2.7.0")
DIRECT_BUILD_VERSION = version.parse("2.9.dev0")


@pytest.mark.skipif(
    not torch.cuda.is_available() or TORCH_VERSION < MINIMUM_TORCH_VERSION,
    reason="CUDA not available or PyTorch version < 2.7",
)
def test_flex_attention_vs_default_backend(vllm_runner):
    """Test that FlexAttention produces the same outputs as the default backend.

    This test compares the outputs from the FlexAttention backend with
    the default backend, ensuring they are similar when using the same seed.
    """
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    seed = 42
    max_tokens = 24
    num_logprobs = 5
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
    ]

    # Run with flex attention
    set_random_seed(seed)
    with vllm_runner(
        model_name,
        runner="generate",
        tensor_parallel_size=1,
        num_gpu_blocks_override=128,
        enforce_eager=True,
        attention_config={"backend": "FLEX_ATTENTION"},
    ) as llm_flex:
        output_flex = llm_flex.generate_greedy_logprobs(
            prompts, max_tokens, num_logprobs
        )

    # Run with default backend
    set_random_seed(seed)
    with vllm_runner(
        model_name,
        runner="generate",
        tensor_parallel_size=1,
        num_gpu_blocks_override=128,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
    ) as llm_default:
        output_default = llm_default.generate_greedy_logprobs(
            prompts, max_tokens, num_logprobs
        )

    check_logprobs_close(
        outputs_0_lst=output_flex,
        outputs_1_lst=output_default,
        name_0="flex",
        name_1="default",
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or TORCH_VERSION < MINIMUM_TORCH_VERSION,
    reason="CUDA not available or PyTorch version < 2.7",
)
def test_encoder_flex_attention_vs_default_backend(vllm_runner):
    """Test that FlexAttention produces the same outputs as the default backend.

    This test compares the outputs from the FlexAttention backend with
    the default backend for encoder models.
    """
    model_name = "BAAI/bge-base-en-v1.5"
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
    ]

    # Run with flex attention
    with vllm_runner(
        model_name,
        runner="pooling",
        dtype=torch.bfloat16,
        tensor_parallel_size=1,
        max_model_len=100,
        enforce_eager=True,
        attention_config={"backend": "FLEX_ATTENTION"},
    ) as llm_flex:
        flex_outputs = llm_flex.embed(prompts)

    # Run with default backend
    with vllm_runner(
        model_name,
        runner="pooling",
        dtype=torch.bfloat16,
        tensor_parallel_size=1,
        max_model_len=100,
        enforce_eager=True,
    ) as llm_default:
        default_outputs = llm_default.embed(prompts)

    check_embeddings_close(
        embeddings_0_lst=flex_outputs,
        embeddings_1_lst=default_outputs,
        name_0="flex",
        name_1="default",
        tol=1e-2,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or TORCH_VERSION < DIRECT_BUILD_VERSION,
    reason="CUDA not available or PyTorch version < 2.7",
)
def test_block_mask_direct_vs_slow_path():
    """Test that direct path block mask is a superset of slow path.

    The direct path may include extra blocks for performance (over-estimation),
    but must include all blocks that the slow path determines are necessary.
    """
    device = torch.device("cuda")

    vllm_config = create_vllm_config(
        model_name="meta-llama/Meta-Llama-3-8B", block_size=16, max_model_len=1024
    )
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)

    # Use a mixed batch that will create groups spanning multiple sequences
    batch_spec = BatchSpec(
        seq_lens=[35, 64, 128, 256], query_lens=[33, 5, 32, 64], name="test_mixed_batch"
    )

    common_attn_metadata = create_common_attn_metadata(
        batch_spec, vllm_config.cache_config.block_size, device
    )

    builder = FlexAttentionMetadataBuilder(kv_cache_spec, [], vllm_config, device)

    metadata_direct = builder.build(
        common_prefix_len=0, common_attn_metadata=common_attn_metadata
    )
    builder.direct_build = False
    metadata_slow = builder.build(
        common_prefix_len=0, common_attn_metadata=common_attn_metadata
    )

    assert metadata_direct.block_mask is not None
    assert metadata_slow.block_mask is not None

    # Extract block indices for comparison, B, H are the same
    direct_indices = metadata_direct.block_mask.kv_indices[0, 0]
    slow_indices = metadata_slow.block_mask.kv_indices[0, 0]
    direct_num = metadata_direct.block_mask.kv_num_blocks[0, 0]
    slow_num = metadata_slow.block_mask.kv_num_blocks[0, 0]

    # main test: every block needed by slow path must be in direct path
    num_groups = direct_num.shape[0]
    all_contained = True
    missing_details = []

    for group_idx in range(num_groups):
        direct_blocks = set(direct_indices[group_idx, : direct_num[group_idx]].tolist())
        slow_blocks = set(slow_indices[group_idx, : slow_num[group_idx]].tolist())

        missing_blocks = slow_blocks - direct_blocks
        if missing_blocks:
            all_contained = False
            missing_details.append(
                f"Group {group_idx}: missing {sorted(missing_blocks)}"
            )

    assert all_contained, (
        "Direct path is missing blocks required by slow path:\n"
        + "\n".join(missing_details)
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or TORCH_VERSION < DIRECT_BUILD_VERSION,
    reason="CUDA not available or PyTorch version < 2.9",
)
def test_cross_batch_block_mask_direct_vs_slow_path():
    """Test that direct path includes slow-path cross-batch peer blocks."""
    device = torch.device("cuda")
    block_size = 4
    seq_lens = torch.tensor([8, 8, 8, 8, 8], dtype=torch.int32, device=device)
    query_lens = torch.tensor([2, 1, 3, 2, 1], dtype=torch.int32, device=device)
    query_start_loc = torch.zeros(len(query_lens) + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = query_lens.cumsum(0)
    num_actual_tokens = int(query_lens.sum().item())

    # Request rows are deliberately not grouped contiguously:
    #   group 0: requests 0 and 2
    #   group 1: requests 1 and 3
    #   request 4: ungrouped traffic
    block_table = torch.tensor(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
        ],
        dtype=torch.int32,
        device=device,
    )
    total_blocks = 16
    token_ids = torch.tensor(
        [
            [11, 12, 99, 14, 15, 16, 99, 18],
            [21, 22, 99, 24, 25, 26, 99, 28],
            [31, 32, 99, 34, 35, 36, 99, 38],
            [41, 42, 99, 44, 45, 46, 99, 48],
            [51, 52, 99, 54, 55, 56, 99, 58],
        ],
        dtype=torch.int32,
        device=device,
    )
    cross_batch = CrossBatchAttentionMetadata(
        enabled=torch.tensor([True, True, True, True, False], device=device),
        group_ids=torch.tensor([0, 1, 0, 1, -1], dtype=torch.int32, device=device),
        replica_ids=torch.tensor([0, 0, 1, 1, -1], dtype=torch.int32, device=device),
        group_members=torch.tensor(
            [[0, 2], [1, 3]], dtype=torch.int32, device=device
        ),
        group_member_mask=torch.tensor(
            [[True, True], [True, True]], device=device
        ),
        virtual_token_ids=torch.tensor(
            [99, 99, 99, 99, 99], dtype=torch.int32, device=device
        ),
        virtual_window_sizes=torch.tensor(
            [4, 4, 4, 4, 4], dtype=torch.int32, device=device
        ),
        token_ids=token_ids,
    )

    common_kwargs = dict(
        causal=True,
        num_actual_tokens=num_actual_tokens,
        max_query_len=int(query_lens.max().item()),
        query_start_loc=query_start_loc,
        max_seq_len=int(seq_lens.max().item()),
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=torch.arange(num_actual_tokens, dtype=torch.int64, device=device),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        total_cache_tokens=total_blocks * block_size,
        block_size=block_size,
        max_possible_sequence_length=int(seq_lens.max().item()),
        num_reqs=len(seq_lens),
        physical_to_logical=physical_to_logical_mapping(
            block_table=block_table,
            seq_lens=seq_lens,
            block_size=block_size,
            total_blocks=total_blocks,
        ),
        decode_offset=seq_lens - query_lens,
        num_blocks_per_seq=((seq_lens + block_size - 1) // block_size).to(torch.int32),
        q_block_size=16,
        kv_block_size=block_size,
        cross_batch_attention_metadata=cross_batch,
    )
    metadata_direct = FlexAttentionMetadata(direct_build=True, **common_kwargs)
    metadata_slow = FlexAttentionMetadata(direct_build=False, **common_kwargs)

    assert metadata_direct.block_mask is not None
    assert metadata_slow.block_mask is not None

    direct_indices = metadata_direct.block_mask.kv_indices[0, 0]
    slow_indices = metadata_slow.block_mask.kv_indices[0, 0]
    direct_num = metadata_direct.block_mask.kv_num_blocks[0, 0]
    slow_num = metadata_slow.block_mask.kv_num_blocks[0, 0]

    missing_details = []
    for group_idx in range(direct_num.shape[0]):
        direct_blocks = set(direct_indices[group_idx, : direct_num[group_idx]].tolist())
        slow_blocks = set(slow_indices[group_idx, : slow_num[group_idx]].tolist())
        missing_blocks = slow_blocks - direct_blocks
        if missing_blocks:
            missing_details.append(
                f"Group {group_idx}: missing {sorted(missing_blocks)}"
            )

    assert not missing_details, (
        "Direct path is missing cross-batch blocks required by slow path:\n"
        + "\n".join(missing_details)
    )


def test_physical_to_logical_mapping_handles_reused_blocks():
    """Regression test: reused physical blocks map to the latest logical block.

    For sliding-window / hybrid attention layers, physical KV-cache blocks can be
    reused over time. The inverse mapping must therefore select the latest
    logical block index for a physical block id.
    """
    # Padding should not make physical block 0 look live.
    block_table = torch.tensor([[6, 0, 0, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([1 * 16], dtype=torch.int32)  # only 1 block valid
    out = physical_to_logical_mapping(
        block_table=block_table, seq_lens=seq_lens, block_size=16, total_blocks=10
    )
    assert out[0, 0].item() == -1
    assert out[0, 6].item() == 0

    # If a physical block id appears multiple times (block reuse), mapping should
    # point to the latest logical block index.
    block_table2 = torch.tensor([[2, 2, 5]], dtype=torch.int32)
    seq_lens2 = torch.tensor([3 * 16], dtype=torch.int32)
    out2 = physical_to_logical_mapping(
        block_table=block_table2, seq_lens=seq_lens2, block_size=16, total_blocks=8
    )
    assert out2[0, 2].item() == 1


if __name__ == "__main__":
    pytest.main([__file__])
