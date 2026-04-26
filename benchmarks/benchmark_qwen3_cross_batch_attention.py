import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path

import torch
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.tokenizers import get_tokenizer
from vllm.v1.cba_profile import log_event


def build_prompt_ids(tokenizer, prompt_len: int, variant: int) -> list[int]:
    text = (
        f"Problem {variant}: Calculate {7 + variant} + {15 + variant}.\n"
        "Solution: We need to reason carefully and give the final answer."
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    pad_id = tokenizer.encode(" ", add_special_tokens=False)[0]
    if len(ids) > prompt_len:
        return ids[:prompt_len]
    return ids + [pad_id] * (prompt_len - len(ids))


def make_sampling_params(
    mode: str,
    num_reqs: int,
    group_size: int,
    max_tokens: int,
    virtual_window_size: int,
) -> list[SamplingParams]:
    params = []
    for i in range(num_reqs):
        extra_args = {}
        if mode == "cross":
            group_idx = i // group_size
            replica_id = i % group_size
            extra_args = {
                "cross_batch_attention": {
                    "enabled": True,
                    "group_id": f"bench-{group_idx}",
                    "replica_id": replica_id,
                    "group_size": group_size,
                    "virtual_token_id": -1,
                    "virtual_window_size": virtual_window_size,
                }
            }
        params.append(
            SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
                ignore_eos=True,
                extra_args=extra_args,
            )
        )
    return params


def run_case(llm, tokenizer, mode, groups, group_size, prompt_len, max_tokens, repeats):
    num_reqs = groups * group_size
    case_fields = {
        "mode": mode,
        "groups": groups,
        "group_size": group_size,
        "num_reqs": num_reqs,
        "prompt_len": prompt_len,
        "max_tokens": max_tokens,
    }
    log_event("bench.case_start", **case_fields)
    prompts = [
        TokensPrompt(prompt_token_ids=build_prompt_ids(tokenizer, prompt_len, i))
        for i in range(num_reqs)
    ]
    params = make_sampling_params(
        mode=mode,
        num_reqs=num_reqs,
        group_size=group_size,
        max_tokens=max_tokens,
        virtual_window_size=prompt_len - 1,
    )

    # Warmup for this exact shape/topology.
    log_event("bench.warmup_start", **case_fields)
    llm.generate(prompts, params, use_tqdm=False)
    torch.cuda.synchronize()
    log_event("bench.warmup_end", **case_fields)

    trials = []
    for trial_idx in range(repeats):
        log_event("bench.trial_start", trial_idx=trial_idx, **case_fields)
        start = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        output_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
        prompt_tokens = num_reqs * prompt_len
        trials.append(
            {
                "elapsed_s": elapsed,
                "output_tokens": output_tokens,
                "prompt_tokens": prompt_tokens,
                "output_tok_s": output_tokens / elapsed,
                "total_tok_s": (prompt_tokens + output_tokens) / elapsed,
            }
        )
        log_event(
            "bench.trial_end",
            trial_idx=trial_idx,
            elapsed_s=elapsed,
            output_tokens=output_tokens,
            output_tok_s=output_tokens / elapsed,
            **case_fields,
        )
    output_rates = [trial["output_tok_s"] for trial in trials]
    total_rates = [trial["total_tok_s"] for trial in trials]
    log_event("bench.case_end", **case_fields)
    return {
        "mode": mode,
        "groups": groups,
        "group_size": group_size,
        "num_reqs": num_reqs,
        "prompt_len": prompt_len,
        "max_tokens": max_tokens,
        "virtual_window_size": prompt_len - 1,
        "trials": trials,
        "output_tok_s_mean": statistics.mean(output_rates),
        "output_tok_s_stdev": statistics.stdev(output_rates) if len(output_rates) > 1 else 0.0,
        "total_tok_s_mean": statistics.mean(total_rates),
        "total_tok_s_stdev": statistics.stdev(total_rates) if len(total_rates) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/qwen3_cba_benchmark.json")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--groups", default="1,2,4,8")
    parser.add_argument("--prompt-lens", default="128,512")
    parser.add_argument("--decode-lens", default="64,256")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--attention-backend", default="FLEX_ATTENTION")
    parser.add_argument("--modes", default="baseline,cross")
    args = parser.parse_args()

    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

    groups_list = [int(x) for x in args.groups.split(",") if x]
    prompt_lens = [int(x) for x in args.prompt_lens.split(",") if x]
    decode_lens = [int(x) for x in args.decode_lens.split(",") if x]
    max_groups = max(groups_list)
    max_prompt_len = max(prompt_lens)
    max_decode_len = max(decode_lens)
    max_num_seqs = max_groups * args.group_size

    tokenizer = get_tokenizer(args.model)
    results = {
        "model": args.model,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "max_num_seqs": max_num_seqs,
        "attention_backend": args.attention_backend,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "cases": [],
    }

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=max_prompt_len + max_decode_len + 8,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        attention_config={"backend": args.attention_backend},
        enable_prefix_caching=False,
        disable_log_stats=True,
    )

    modes = [mode for mode in args.modes.split(",") if mode]
    for prompt_len in prompt_lens:
        for max_tokens in decode_lens:
            for groups in groups_list:
                for mode in modes:
                    case = run_case(
                        llm=llm,
                        tokenizer=tokenizer,
                        mode=mode,
                        groups=groups,
                        group_size=args.group_size,
                        prompt_len=prompt_len,
                        max_tokens=max_tokens,
                        repeats=args.repeats,
                    )
                    results["cases"].append(case)
                    print(json.dumps({k: v for k, v in case.items() if k != "trials"}))
                    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
                    gc.collect()
                    torch.cuda.empty_cache()

    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
