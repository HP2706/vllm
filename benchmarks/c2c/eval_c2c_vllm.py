"""Evaluate C2C-in-vLLM on OpenBookQA against two running vLLM servers.

Reuses the C2C repo's prompt builder and answer extraction so numbers are
comparable to the offline unified_evaluator results.

Flow per question (fused mode):
  1. POST prompt to producer (sharer, max_tokens=1)  -> stages sharer KV
  2. POST prompt to consumer (max_tokens=1)          -> transfer + fusion
  3. POST prompt to consumer (max_tokens=64)         -> prefix-cache hits the
     fused blocks; all decode tokens attend to fused KV
Baseline mode skips step 1 (no manifest -> no fusion).

Run in the C2C venv (needs rosetta + datasets):
  python eval_c2c_vllm.py run --mode baseline --limit 100
  python eval_c2c_vllm.py run --mode fused --limit 100
"""

import json
import os
import time
from typing import Any

import fire
import requests
from datasets import load_dataset
from transformers import AutoTokenizer

from rosetta.utils.evaluate import build_prompt, extract_answer_from_content

PRODUCER_PORT = 8100
CONSUMER_PORT = 8200
PRODUCER_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CONSUMER_MODEL = "Qwen/Qwen3-0.6B"
OUT_DIR = os.path.expanduser("~/bench_logs")


def _completion(
    port: int, model: str, prompt: str | list[int], max_tokens: int
) -> tuple[str, float]:
    # Both servers receive the receiver-tokenized ids, matching the offline
    # RosettaModel setup (alignment off: sharer consumes receiver token ids).
    t0 = time.perf_counter()
    r = requests.post(
        f"http://localhost:{port}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["text"], (time.perf_counter() - t0) * 1000.0


def _format_question(example: dict[str, Any]) -> str:
    choices = ""
    raw = example["choices"]
    for i, text in enumerate(raw["text"]):
        choices += f"{chr(65 + i)}. {text}\n"
    return build_prompt(
        dataset="mmlu-redux",
        locale="",
        question=example["question_stem"],
        choices=choices,
        use_cot=False,
        use_template=True,
    )


def run(mode: str = "fused", limit: int = 100, max_tokens: int = 64) -> None:
    assert mode in ("fused", "baseline")
    ds = load_dataset("openbookqa", "main")["test"]
    n = min(limit, len(ds))
    tok = AutoTokenizer.from_pretrained(CONSUMER_MODEL)

    rows: list[dict[str, Any]] = []
    correct = 0
    for i in range(n):
        ex = ds[i]
        question_prompt = _format_question(ex)
        prompt_ids: list[int] = tok.apply_chat_template(
            [{"role": "user", "content": question_prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        timings: dict[str, float] = {}
        if mode == "fused":
            _, timings["producer_ms"] = _completion(
                PRODUCER_PORT, PRODUCER_MODEL, prompt_ids, 1
            )
            _, timings["fusion_req_ms"] = _completion(
                CONSUMER_PORT, CONSUMER_MODEL, prompt_ids, 1
            )
        answer_text, timings["answer_ms"] = _completion(
            CONSUMER_PORT, CONSUMER_MODEL, prompt_ids, max_tokens
        )
        predicted = extract_answer_from_content(answer_text)
        gold = ex["answerKey"]
        ok = predicted is not None and predicted.strip().upper().startswith(gold)
        correct += int(ok)
        rows.append(
            {
                "i": i,
                "gold": gold,
                "predicted": predicted,
                "correct": ok,
                "text": answer_text[:200],
                **timings,
            }
        )
        if (i + 1) % 20 == 0:
            print(f"[{i + 1}/{n}] running accuracy: {correct / (i + 1):.3f}")

    acc = correct / n
    out = {
        "mode": mode,
        "n": n,
        "accuracy": acc,
        "rows": rows,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"eval_vllm_{mode}_{n}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nmode={mode} n={n} accuracy={acc:.4f}")
    print(f"written to {path}")


if __name__ == "__main__":
    fire.Fire({"run": run})
