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
import os
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


def save_tensor(path: str, data: dict[str, Any]) -> None:
    import torch

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)


def make_sampling_params(args: argparse.Namespace) -> list[Any]:
    from vllm import SamplingParams

    params = []
    for replica_id in range(args.group_size):
        extra_args = None
        if args.cross_batch:
            extra_args = {
                "cross_batch_attention": {
                    "enabled": True,
                    "group_id": "trained-probe",
                    "replica_id": replica_id,
                    "group_size": args.group_size,
                    "virtual_token_id": -1,
                    "virtual_window_size": args.virtual_window_size,
                }
            }
        params.append(
            SamplingParams(
                temperature=0.0,
                max_tokens=1,
                logprobs=getattr(args, "top_k", None),
                extra_args=extra_args,
            )
        )
    return params


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
            "cross_batch_attend": args.cross_batch,
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
        "cross_batch": args.cross_batch,
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


def dump_hf_logits(args: argparse.Namespace) -> None:
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
            "cross_batch_attend": args.cross_batch,
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
        logits = model(input_ids=input_ids).logits[:, -1, :].float().cpu()

    logprobs = torch.log_softmax(logits, dim=-1)
    save_tensor(
        args.output,
        {
            "engine": "hf",
            "base_model": args.base_model,
            "adapter": args.adapter,
            "cross_batch": args.cross_batch,
            "prompt": args.prompt,
            "prompt_ids": prompt_ids,
            "group_size": args.group_size,
            "virtual_window_size": args.virtual_window_size,
            "vocab_size": logits.shape[-1],
            "logits": logits,
            "logprobs": logprobs,
        },
    )
    print(f"wrote HF logits {tuple(logits.shape)} to {args.output}")


def run_vllm(args: argparse.Namespace) -> None:
    from vllm import LLM, TokensPrompt
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
    sampling_params = make_sampling_params(args)
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
        "cross_batch": args.cross_batch,
        "prompt": args.prompt,
        "prompt_ids": prompt_ids,
        "generated": outputs[0].outputs[0].text,
        "generated_token_ids": list(outputs[0].outputs[0].token_ids),
        "top": top,
    }
    write_json(args.output, result)
    print(json.dumps(result["top"][: min(8, args.top_k)], indent=2))


def dump_vllm_logits(args: argparse.Namespace) -> None:
    import torch

    from vllm import LLM, TokensPrompt
    from vllm.config import VllmConfig
    from vllm.lora.request import LoRARequest
    from vllm.tokenizers import get_tokenizer
    from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor

    class DumpLogitsProcessor(LogitsProcessor):

        def __init__(
            self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
        ):
            self.path = os.environ["VLLM_CBA_DUMP_LOGITS_PATH"]
            self.call_idx = 0

        def is_argmax_invariant(self) -> bool:
            return False

        def update_state(self, batch_update: BatchUpdate | None):
            return None

        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            if self.call_idx == 0:
                save_tensor(
                    self.path,
                    {
                        "engine": "vllm",
                        "call_idx": self.call_idx,
                        "logits": logits.detach().float().cpu(),
                    },
                )
            self.call_idx += 1
            return logits

    tokenizer = get_tokenizer(args.base_model)
    prompt_ids = build_prompt_ids(tokenizer, args.prompt, args.prompt_len)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    os.environ["VLLM_CBA_DUMP_LOGITS_PATH"] = str(output_path)

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
        logits_processors=[DumpLogitsProcessor],
    )
    lora = LoRARequest("trained_cross_batch", 1, args.adapter)
    llm.llm_engine.add_lora(lora)
    sampling_params = make_sampling_params(args)
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids) for _ in range(args.group_size)],
        sampling_params,
        lora_request=lora,
        use_tqdm=False,
    )
    dumped = torch.load(output_path, map_location="cpu")
    logits = dumped["logits"]
    dumped.update(
        {
            "base_model": args.base_model,
            "adapter": args.adapter,
            "cross_batch": args.cross_batch,
            "prompt": args.prompt,
            "prompt_ids": prompt_ids,
            "group_size": args.group_size,
            "virtual_window_size": args.virtual_window_size,
            "vocab_size": logits.shape[-1],
            "generated": outputs[0].outputs[0].text,
            "generated_token_ids": list(outputs[0].outputs[0].token_ids),
            "logprobs": torch.log_softmax(logits, dim=-1),
        }
    )
    save_tensor(args.output, dumped)
    print(f"wrote vLLM logits {tuple(logits.shape)} to {args.output}")


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


def _tensor_metrics(diff: Any) -> dict[str, float]:
    import torch

    flat = diff.abs().flatten()
    quantiles = torch.quantile(
        flat, torch.tensor([0.5, 0.9, 0.99, 0.999], device=flat.device)
    )
    return {
        "max_abs": float(flat.max()),
        "mean_abs": float(flat.mean()),
        "rmse": float(torch.sqrt(torch.mean(diff.float().pow(2)))),
        "p50_abs": float(quantiles[0]),
        "p90_abs": float(quantiles[1]),
        "p99_abs": float(quantiles[2]),
        "p999_abs": float(quantiles[3]),
    }


def compare_logits(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer

    hf = torch.load(args.hf_logits, map_location="cpu")
    vllm = torch.load(args.vllm_logits, map_location="cpu")
    hf_logits = hf["logits"].float()
    vllm_logits = vllm["logits"].float()
    shared_batch = min(hf_logits.shape[0], vllm_logits.shape[0])
    shared_vocab = min(hf_logits.shape[1], vllm_logits.shape[1])
    hf_logits = hf_logits[:shared_batch, :shared_vocab]
    vllm_logits = vllm_logits[:shared_batch, :shared_vocab]
    hf_logprobs = torch.log_softmax(hf_logits, dim=-1)
    vllm_logprobs = torch.log_softmax(vllm_logits, dim=-1)

    logits_diff = vllm_logits - hf_logits
    logprobs_diff = vllm_logprobs - hf_logprobs
    max_batch, max_token = divmod(
        int(logprobs_diff.abs().argmax()), logprobs_diff.shape[1]
    )
    tokenizer = AutoTokenizer.from_pretrained(hf.get("base_model", DEFAULT_BASE_MODEL))
    topk_overlap = {}
    for k in args.top_k:
        limit = min(k, shared_vocab)
        hf_top = torch.topk(hf_logprobs, limit, dim=-1).indices
        vllm_top = torch.topk(vllm_logprobs, limit, dim=-1).indices
        overlaps = []
        for batch_idx in range(shared_batch):
            overlaps.append(
                len(set(hf_top[batch_idx].tolist()) & set(vllm_top[batch_idx].tolist()))
                / limit
            )
        topk_overlap[str(k)] = {
            "mean": float(sum(overlaps) / len(overlaps)),
            "per_batch": overlaps,
        }

    per_batch = []
    for batch_idx in range(shared_batch):
        per_batch.append(
            {
                "batch": batch_idx,
                "hf_top1": int(hf_logprobs[batch_idx].argmax()),
                "vllm_top1": int(vllm_logprobs[batch_idx].argmax()),
                "top1_match": bool(
                    hf_logprobs[batch_idx].argmax()
                    == vllm_logprobs[batch_idx].argmax()
                ),
                "logits": _tensor_metrics(logits_diff[batch_idx]),
                "logprobs": _tensor_metrics(logprobs_diff[batch_idx]),
            }
        )

    result = {
        "hf_shape": list(hf["logits"].shape),
        "vllm_shape": list(vllm["logits"].shape),
        "compared_shape": [shared_batch, shared_vocab],
        "ignored_hf_vocab": int(hf["logits"].shape[1] - shared_vocab),
        "ignored_vllm_vocab": int(vllm["logits"].shape[1] - shared_vocab),
        "hf_prompt_ids_match_vllm": hf.get("prompt_ids") == vllm.get("prompt_ids"),
        "logits": _tensor_metrics(logits_diff),
        "logprobs": _tensor_metrics(logprobs_diff),
        "topk_overlap": topk_overlap,
        "max_logprob_abs_diff": {
            "batch": int(max_batch),
            "token_id": int(max_token),
            "token": tokenizer.decode([int(max_token)]),
            "hf_logprob": float(hf_logprobs[max_batch, max_token]),
            "vllm_logprob": float(vllm_logprobs[max_batch, max_token]),
            "abs_diff": float(logprobs_diff[max_batch, max_token].abs()),
        },
        "per_batch": per_batch,
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
        subparser.add_argument(
            "--disable-cross-batch",
            action="store_false",
            dest="cross_batch",
            help="Run the same probe without cross-batch attention.",
        )
        subparser.set_defaults(cross_batch=True)

    hf_parser = subparsers.add_parser("hf")
    add_common(hf_parser)
    hf_parser.add_argument("--parent-repo", default=DEFAULT_PARENT_REPO)
    hf_parser.set_defaults(func=run_hf)

    dump_hf_parser = subparsers.add_parser("dump-hf-logits")
    add_common(dump_hf_parser)
    dump_hf_parser.add_argument("--parent-repo", default=DEFAULT_PARENT_REPO)
    dump_hf_parser.set_defaults(func=dump_hf_logits)

    vllm_parser = subparsers.add_parser("vllm")
    add_common(vllm_parser)
    vllm_parser.add_argument("--max-lora-rank", type=int, default=16)
    vllm_parser.add_argument("--max-model-len", type=int, default=192)
    vllm_parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    vllm_parser.set_defaults(func=run_vllm)

    dump_vllm_parser = subparsers.add_parser("dump-vllm-logits")
    add_common(dump_vllm_parser)
    dump_vllm_parser.add_argument("--max-lora-rank", type=int, default=16)
    dump_vllm_parser.add_argument("--max-model-len", type=int, default=192)
    dump_vllm_parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.45
    )
    dump_vllm_parser.set_defaults(func=dump_vllm_logits)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--hf-json", required=True)
    compare_parser.add_argument("--vllm-json", required=True)
    compare_parser.add_argument("--output")
    compare_parser.set_defaults(func=run_compare)

    compare_logits_parser = subparsers.add_parser("compare-logits")
    compare_logits_parser.add_argument("--hf-logits", required=True)
    compare_logits_parser.add_argument("--vllm-logits", required=True)
    compare_logits_parser.add_argument(
        "--top-k", type=int, nargs="+", default=[1, 5, 20, 100]
    )
    compare_logits_parser.add_argument("--output")
    compare_logits_parser.set_defaults(func=compare_logits)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
