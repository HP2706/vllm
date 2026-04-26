# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from tests.utils import large_gpu_mark, single_gpu_only


MODEL = "Qwen/Qwen3-0.6B"
PROMPTS = [
    "Solve briefly: 2 + 2 = reasoning",
    "Solve briefly: 3 + 5 = reasoning",
]
VIRTUAL_WINDOW_SIZE = 10
SELECTED_TOKEN_IDS = [3019, 1882, 198, 13, 30]


def _repo_root() -> Path:
    # tests/v1/<this file> -> vllm repo -> parent project repo
    return Path(__file__).resolve().parents[2].parent


def _load_hf_reference_model():
    root = _repo_root()
    if not (root / "src/modeling_qwen3_batch_parscale.py").exists():
        pytest.skip("HF batch-parscale reference source is not available")
    sys.path.insert(0, str(root))
    pytest.importorskip("accelerate")

    from src.modeling_qwen3_batch_parscale import Qwen3BatchParScaleForCausalLM

    model, _ = Qwen3BatchParScaleForCausalLM.from_qwen_model(
        MODEL,
        config_kwargs={
            "n_batches": 2,
            "cross_batch_attend": True,
            "noise_method": "none",
            "virtual_window_size": VIRTUAL_WINDOW_SIZE,
        },
        dtype=torch.bfloat16,
        attn_implementation="flex_attention",
    )
    return model.eval().to("cuda")


def _hf_reference_logprobs(tokenizer: AutoTokenizer) -> torch.Tensor:
    model = _load_hf_reference_model()
    input_ids = tokenizer(PROMPTS, return_tensors="pt", padding=True).input_ids.to(
        "cuda"
    )
    attention_group_ids = torch.zeros(len(PROMPTS), dtype=torch.long, device="cuda")

    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_group_ids=attention_group_ids,
                num_logits_to_keep=1,
                use_cache=False,
            )
            return F.log_softmax(outputs.logits[:, -1, :].float(), dim=-1).cpu()
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _vllm_cross_batch_logprobs(tokenizer: AutoTokenizer) -> list[dict[int, float]]:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"

    from vllm import LLM, SamplingParams

    virtual_token_id = tokenizer.encode(" reasoning", add_special_tokens=False)[0]
    sampling_params = [
        SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=10,
            extra_args={
                "cross_batch_attention": {
                    "enabled": True,
                    "group_id": "hf-reference",
                    "replica_id": replica_id,
                    "group_size": 2,
                    "virtual_token_id": virtual_token_id,
                    "virtual_window_size": VIRTUAL_WINDOW_SIZE,
                }
            },
        )
        for replica_id in range(2)
    ]
    llm = LLM(
        model=MODEL,
        max_model_len=128,
        gpu_memory_utilization=0.35,
        enforce_eager=True,
        trust_remote_code=True,
        attention_config={"backend": "FLEX_ATTENTION"},
        disable_log_stats=True,
    )
    outputs = llm.generate(PROMPTS, sampling_params, use_tqdm=False)
    return [
        {
            token_id: logprob.logprob
            for token_id, logprob in output.outputs[0].logprobs[0].items()
        }
        for output in outputs
    ]


@pytest.mark.slow_test
@single_gpu_only
@large_gpu_mark(min_gb=16)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qwen3_cross_batch_logits_match_hf_reference():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    hf_logprobs = _hf_reference_logprobs(tokenizer)
    vllm_logprobs = _vllm_cross_batch_logprobs(tokenizer)

    for row_idx, row_logprobs in enumerate(vllm_logprobs):
        hf_top5 = set(torch.topk(hf_logprobs[row_idx], 5).indices.tolist())
        assert hf_top5 == set(row_logprobs) & hf_top5

        for token_id in SELECTED_TOKEN_IDS:
            assert token_id in row_logprobs
            assert row_logprobs[token_id] == pytest.approx(
                float(hf_logprobs[row_idx, token_id]), abs=0.15
            )
