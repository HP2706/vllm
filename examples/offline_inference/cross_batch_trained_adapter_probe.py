# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in probe for a trained Qwen3 cross-batch LoRA checkpoint.

Run `hf` mode with the parent repo environment that has `peft` and the custom
HF model source. Run `vllm` mode with the vLLM fork environment. Then run
`compare` mode against the two JSON files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_ADAPTER = (
    "/home/ubuntu/parallel_reasoning/experiments/finetune/checkpoints/"
    "nb4_nr0.5_nv4_cbad0.1_2026-04-26-12-33-30/"
    "step_0001107_tokens_9M/adapter"
)
DEFAULT_PARENT_REPO = "/home/ubuntu/parallel_reasoning"
DEFAULT_PROMPT = "Problem: What is 7 + 15?\nSolution:"


def build_prompt_ids(tokenizer: Any, prompt: str, prompt_len: int) -> list[int]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > prompt_len:
        raise ValueError(
            f"Prompt already has {len(ids)} tokens, longer than --prompt-len "
            f"{prompt_len}"
        )
    pad_id = tokenizer.encode(" ", add_special_tokens=False)[0]
    return ids + [pad_id] * (prompt_len - len(ids))


def write_json(path: str, data: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2) + "\n")


def run_hf(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    sys.path.insert(0, args.parent_repo)
    from src.modeling_qwen3_batch_parscale import Qwen3BatchParScaleForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_ids = build_prompt_ids(tokenizer, args.prompt, args.prompt_len)
    model, _ = Qwen3BatchParScaleForCausalLM.from_qwen_model(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flex_attention",
        config_kwargs={
            "n_batches": args.group_size,
            "parscale_n": args.parscale_n,
            "cross_batch_attend": True,
            "noise_ratio": 0.0,
            "noise_method": "none",
            "virtual_window_size": args.virtual_window_size,
            "cross_batch_attention_dropout": 0.0,
        },
    )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval().cuda()

    input_ids = torch.tensor(
        [prompt_ids] * args.group_size, dtype=torch.long, device="cuda"
    )
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        logits = model(input_ids=input_ids).logits[:, -1, :].float()
    logprobs = torch.log_softmax(logits, dim=-1)
    top = torch.topk(logprobs[0], args.top_k)
    result = {
        "engine": "hf",
        "base_model": args.base_model,
        "adapter": args.adapter,
        "prompt": args.prompt,
        "prompt_ids": prompt_ids,
        "top": [
            {
                "id": int(token_id),
                "logprob": float(logprob),
                "token": tokenizer.decode([int(token_id)]),
            }
            for logprob, token_id in zip(top.values, top.indices)
        ],
    }
    write_json(args.output, result)
    print(json.dumps(result["top"][: min(8, args.top_k)], indent=2))


def run_vllm(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.lora.request import LoRARequest
    from vllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer(args.base_model)
    prompt_ids = build_prompt_ids(tokenizer, args.prompt, args.prompt_len)
    llm = LLM(
        model=args.base_model,
        dtype="bfloat16",
        enable_lora=True,
        max_loras=1,
        max_lora_rank=args.max_lora_rank,
        max_model_len=args.max_model_len,
        max_num_seqs=args.group_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        attention_config={"backend": "FLEX_ATTENTION"},
    )
    lora = LoRARequest("trained_cross_batch", 1, args.adapter)
    llm.llm_engine.add_lora(lora)
    sampling_params = [
        SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=args.top_k,
            extra_args={
                "cross_batch_attention": {
                    "enabled": True,
                    "group_id": "trained-probe",
                    "replica_id": replica_id,
                    "group_size": args.group_size,
                    "virtual_token_id": -1,
                    "virtual_window_size": args.virtual_window_size,
                }
            },
        )
        for replica_id in range(args.group_size)
    ]
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids) for _ in range(args.group_size)],
        sampling_params,
        lora_request=lora,
        use_tqdm=False,
    )
    first_logprobs = outputs[0].outputs[0].logprobs[0]
    top = sorted(
        [
            {
                "id": int(token_id),
                "logprob": float(logprob.logprob),
                "token": logprob.decoded_token,
            }
            for token_id, logprob in first_logprobs.items()
        ],
        key=lambda row: row["logprob"],
        reverse=True,
    )
    result = {
        "engine": "vllm",
        "base_model": args.base_model,
        "adapter": args.adapter,
        "prompt": args.prompt,
        "prompt_ids": prompt_ids,
        "generated": outputs[0].outputs[0].text,
        "generated_token_ids": list(outputs[0].outputs[0].token_ids),
        "top": top,
    }
    write_json(args.output, result)
    print(json.dumps(result["top"][: min(8, args.top_k)], indent=2))


def run_compare(args: argparse.Namespace) -> None:
    hf = json.loads(Path(args.hf_json).read_text())
    vllm = json.loads(Path(args.vllm_json).read_text())
    hf_by_id = {row["id"]: row for row in hf["top"]}
    vllm_by_id = {row["id"]: row for row in vllm["top"]}
    common_ids = [row["id"] for row in hf["top"] if row["id"] in vllm_by_id]
    rows = []
    for token_id in common_ids:
        hf_row = hf_by_id[token_id]
        vllm_row = vllm_by_id[token_id]
        rows.append(
            {
                "id": token_id,
                "token": hf_row["token"],
                "hf_logprob": hf_row["logprob"],
                "vllm_logprob": vllm_row["logprob"],
                "abs_diff": abs(hf_row["logprob"] - vllm_row["logprob"]),
            }
        )
    result = {
        "hf_top1": hf["top"][0],
        "vllm_top1": vllm["top"][0],
        "top1_token_match": hf["top"][0]["id"] == vllm["top"][0]["id"],
        "common_top_tokens": rows,
    }
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
        subparser.add_argument("--adapter", default=DEFAULT_ADAPTER)
        subparser.add_argument("--prompt", default=DEFAULT_PROMPT)
        subparser.add_argument("--prompt-len", type=int, default=128)
        subparser.add_argument("--group-size", type=int, default=4)
        subparser.add_argument("--parscale-n", type=int, default=4)
        subparser.add_argument("--virtual-window-size", type=int, default=127)
        subparser.add_argument("--top-k", type=int, default=20)
        subparser.add_argument("--output", required=True)

    hf_parser = subparsers.add_parser("hf")
    add_common(hf_parser)
    hf_parser.add_argument("--parent-repo", default=DEFAULT_PARENT_REPO)
    hf_parser.set_defaults(func=run_hf)

    vllm_parser = subparsers.add_parser("vllm")
    add_common(vllm_parser)
    vllm_parser.add_argument("--max-lora-rank", type=int, default=16)
    vllm_parser.add_argument("--max-model-len", type=int, default=192)
    vllm_parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    vllm_parser.set_defaults(func=run_vllm)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--hf-json", required=True)
    compare_parser.add_argument("--vllm-json", required=True)
    compare_parser.add_argument("--output")
    compare_parser.set_defaults(func=run_compare)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
