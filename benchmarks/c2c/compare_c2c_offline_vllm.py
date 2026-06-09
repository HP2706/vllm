"""Compare offline Rosetta C2C against vLLM C2C on the same prompts.

The paper's largest released-fuser gain is on ARC-Challenge for
Qwen3-4B-Base -> Qwen3-0.6B. This harness runs the offline Rosetta wrapper and
the vLLM connector path on identical ARC-C prompts, then reports output and
answer agreement. It is intended as a regression test: the vLLM path should
match the offline reference closely before we trust aggregate accuracies.
"""

import gc
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fire
import requests
import torch
from datasets import load_dataset
from transformers.cache_utils import DynamicCache
from transformers import AutoTokenizer

import rosetta.model.wrapper as rosetta_wrapper
from rosetta.utils.evaluate import (
    build_prompt,
    extract_answer_from_content,
    load_rosetta_model,
)


@dataclass(frozen=True)
class ArcPrompt:
    index: int
    prompt: str
    gold: str


def _arc_prompt(example: dict[str, Any], index: int) -> ArcPrompt:
    raw_choices = example["choices"]
    choices_text = ""
    for i, text in enumerate(raw_choices["text"]):
        choices_text += f"{chr(65 + i)}. {text}\n"
    prompt = build_prompt(
        dataset="mmlu-redux",
        locale="",
        question=example["question"],
        choices=choices_text,
        use_cot=False,
        use_template=True,
    )
    return ArcPrompt(index=index, prompt=prompt, gold=example["answerKey"])


def _load_arc(limit: int) -> list[ArcPrompt]:
    dataset = load_dataset("ai2_arc", "ARC-Challenge")["test"]
    return [_arc_prompt(dataset[i], i) for i in range(min(limit, len(dataset)))]


def _prediction(text: str) -> str | None:
    pred = extract_answer_from_content(text)
    if pred is None:
        return None
    return pred.strip().upper()[:1]


def _is_correct(prediction: str | None, gold: str) -> bool:
    return prediction is not None and prediction == gold.strip().upper()[:1]


def _clone_dynamic_cache(cache: DynamicCache) -> DynamicCache:
    legacy_cache = cache.to_legacy_cache()
    cloned = [(key.clone().detach(), value.clone().detach()) for key, value in legacy_cache]
    return DynamicCache.from_legacy_cache(cloned)


def _dynamic_key_cache(cache: DynamicCache) -> list[torch.Tensor]:
    return [key for key, _ in cache.to_legacy_cache()]


def _dynamic_value_cache(cache: DynamicCache) -> list[torch.Tensor]:
    return [value for _, value in cache.to_legacy_cache()]


DynamicCache.key_cache = property(_dynamic_key_cache)  # type: ignore[attr-defined]
DynamicCache.value_cache = property(_dynamic_value_cache)  # type: ignore[attr-defined]
rosetta_wrapper.clone_kv_cache = _clone_dynamic_cache


def _rosetta_config(
    receiver_model: str,
    producer_model: str,
    fuser_dir: str,
    out_dir: str,
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "model": {
            "model_name": "Rosetta",
            "rosetta_config": {
                "base_model": receiver_model,
                "teacher_model": producer_model,
                "is_do_alignment": False,
                "alignment_strategy": "longest",
                "checkpoints_dir": fuser_dir,
            },
            "generation_config": {
                "do_sample": False,
                "max_new_tokens": max_tokens,
            },
        },
        "output": {"output_dir": out_dir},
        "eval": {
            "dataset": "ai2-arc",
            "gpu_ids": [0],
            "answer_method": "generate",
            "use_cot": False,
            "use_template": True,
            "sample_interval": 1,
        },
    }


def run_offline(
    limit: int = 20,
    max_tokens: int = 64,
    receiver_model: str = "Qwen/Qwen3-0.6B",
    producer_model: str = "Qwen/Qwen3-4B-Base",
    fuser_dir: str = "",
    out_path: str = "~/bench_logs/c2c_offline_arc.json",
) -> None:
    assert fuser_dir, "Pass --fuser-dir pointing at the C2C fuser final directory"
    prompts = _load_arc(limit)
    path = Path(os.path.expanduser(out_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    config = _rosetta_config(
        receiver_model=receiver_model,
        producer_model=producer_model,
        fuser_dir=fuser_dir,
        out_dir=str(path.parent),
        max_tokens=max_tokens,
    )
    model, tokenizer = load_rosetta_model(config["model"], config["eval"], device)
    rows: list[dict[str, Any]] = []
    correct = 0
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt.prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokenized = tokenizer(text, return_tensors="pt").to(device)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        position_ids = attention_mask.long().cumsum(-1) - 1
        instruction_length = input_ids.shape[1] - 1
        kv_cache_index = [
            torch.tensor([1, 0], dtype=torch.long)
            .repeat(instruction_length, 1)
            .unsqueeze(0)
            .to(device),
            torch.tensor([-1, 0], dtype=torch.long)
            .repeat(1, 1)
            .unsqueeze(0)
            .to(device),
        ]
        t0 = time.perf_counter()
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache_index=kv_cache_index,
            do_sample=False,
            max_new_tokens=max_tokens,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        generated = output_ids[0][input_ids.shape[1] :]
        text_out = tokenizer.decode(generated, skip_special_tokens=True).strip("\n")
        pred = _prediction(text_out)
        ok = _is_correct(pred, prompt.gold)
        correct += int(ok)
        rows.append(
            {
                "i": prompt.index,
                "gold": prompt.gold,
                "predicted": pred,
                "correct": ok,
                "text": text_out,
                "latency_ms": latency_ms,
            }
        )
    out = {
        "mode": "offline_rosetta",
        "dataset": "ai2_arc/ARC-Challenge",
        "receiver_model": receiver_model,
        "producer_model": producer_model,
        "fuser_dir": fuser_dir,
        "n": len(prompts),
        "accuracy": correct / len(prompts),
        "rows": rows,
    }
    path.write_text(json.dumps(out, indent=2))
    print(f"offline accuracy={out['accuracy']:.4f} n={out['n']} written={path}")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def _completion(
    port: int,
    model: str,
    prompt: str | list[int],
    max_tokens: int,
) -> tuple[str, float]:
    t0 = time.perf_counter()
    response = requests.post(
        f"http://localhost:{port}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["text"], (time.perf_counter() - t0) * 1000.0


def _kv_config(role: str, extra: dict[str, Any]) -> str:
    return json.dumps(
        {
            "kv_connector": "C2CConnector",
            "kv_role": role,
            "kv_connector_extra_config": extra,
        }
    )


def _wait_port(port: int, timeout_s: float = 300.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2.0)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(3)
    raise TimeoutError(f"vLLM server on port {port} did not become ready")


def _terminate(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
    for process in processes:
        if process.poll() is None:
            process.wait(timeout=30)


def run_vllm(
    limit: int = 20,
    max_tokens: int = 64,
    receiver_model: str = "Qwen/Qwen3-0.6B",
    producer_model: str = "Qwen/Qwen3-4B-Base",
    fuser_dir: str = "",
    out_path: str = "~/bench_logs/c2c_vllm_arc.json",
    log_dir: str = "~/bench_logs/c2c_vllm_arc_servers",
    producer_gpu: int = 0,
    consumer_gpu: int = 1,
    producer_port: int = 8100,
    consumer_port: int = 8200,
    c2c_port: int = 8077,
) -> None:
    assert fuser_dir, "Pass --fuser-dir pointing at the C2C fuser final directory"
    prompts = _load_arc(limit)
    out = Path(os.path.expanduser(out_path))
    logs = Path(os.path.expanduser(log_dir))
    timing_dir = logs / "timings"
    out.parent.mkdir(parents=True, exist_ok=True)
    timing_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(receiver_model)
    producer_extra = {
        "c2c_transport": "tcp",
        "c2c_bind_host": "127.0.0.1",
        "c2c_host": "127.0.0.1",
        "c2c_port": c2c_port,
        "c2c_timing_dir": str(timing_dir),
    }
    consumer_extra = {
        "c2c_transport": "tcp",
        "c2c_host": "127.0.0.1",
        "c2c_port": c2c_port,
        "c2c_timing_dir": str(timing_dir),
        "c2c_checkpoints_dir": fuser_dir,
    }
    common = [
        "--gpu-memory-utilization",
        "0.45",
        "--max-model-len",
        "16384",
        "--max-num-batched-tokens",
        "16384",
        "--enforce-eager",
    ]
    prod_env = os.environ.copy()
    prod_env["CUDA_VISIBLE_DEVICES"] = str(producer_gpu)
    cons_env = os.environ.copy()
    cons_env["CUDA_VISIBLE_DEVICES"] = str(consumer_gpu)
    processes: list[subprocess.Popen[bytes]] = []
    producer_log = open(logs / "producer.txt", "wb")
    consumer_log = open(logs / "consumer.txt", "wb")
    try:
        processes.append(
            subprocess.Popen(
                [
                    "vllm",
                    "serve",
                    producer_model,
                    "--port",
                    str(producer_port),
                    "--kv-transfer-config",
                    _kv_config("kv_producer", producer_extra),
                    *common,
                ],
                env=prod_env,
                stdout=producer_log,
                stderr=subprocess.STDOUT,
            )
        )
        processes.append(
            subprocess.Popen(
                [
                    "vllm",
                    "serve",
                    receiver_model,
                    "--port",
                    str(consumer_port),
                    "--kv-transfer-config",
                    _kv_config("kv_consumer", consumer_extra),
                    *common,
                ],
                env=cons_env,
                stdout=consumer_log,
                stderr=subprocess.STDOUT,
            )
        )
        _wait_port(producer_port)
        _wait_port(consumer_port)
        rows: list[dict[str, Any]] = []
        correct = 0
        for prompt in prompts:
            prompt_ids: list[int] = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.prompt}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            _, producer_ms = _completion(producer_port, producer_model, prompt_ids, 1)
            _, fusion_ms = _completion(consumer_port, receiver_model, prompt_ids, 1)
            text_out, answer_ms = _completion(
                consumer_port, receiver_model, prompt_ids, max_tokens
            )
            pred = _prediction(text_out)
            ok = _is_correct(pred, prompt.gold)
            correct += int(ok)
            rows.append(
                {
                    "i": prompt.index,
                    "gold": prompt.gold,
                    "predicted": pred,
                    "correct": ok,
                    "text": text_out,
                    "producer_ms": producer_ms,
                    "fusion_ms": fusion_ms,
                    "answer_ms": answer_ms,
                }
            )
    finally:
        _terminate(processes)
        producer_log.close()
        consumer_log.close()
    result = {
        "mode": "vllm_c2c_tcp_post_write",
        "dataset": "ai2_arc/ARC-Challenge",
        "receiver_model": receiver_model,
        "producer_model": producer_model,
        "fuser_dir": fuser_dir,
        "n": len(prompts),
        "accuracy": correct / len(prompts),
        "rows": rows,
    }
    out.write_text(json.dumps(result, indent=2))
    print(f"vllm accuracy={result['accuracy']:.4f} n={result['n']} written={out}")


def compare(
    offline_path: str = "~/bench_logs/c2c_offline_arc.json",
    vllm_path: str = "~/bench_logs/c2c_vllm_arc.json",
    out_path: str = "~/bench_logs/c2c_arc_comparison.json",
) -> None:
    offline = json.loads(Path(os.path.expanduser(offline_path)).read_text())
    vllm = json.loads(Path(os.path.expanduser(vllm_path)).read_text())
    offline_rows = {row["i"]: row for row in offline["rows"]}
    vllm_rows = {row["i"]: row for row in vllm["rows"]}
    ids = sorted(set(offline_rows) & set(vllm_rows))
    paired: list[dict[str, Any]] = []
    exact_text = 0
    same_pred = 0
    same_correct = 0
    for idx in ids:
        off = offline_rows[idx]
        vlm = vllm_rows[idx]
        text_match = off["text"] == vlm["text"]
        pred_match = off["predicted"] == vlm["predicted"]
        corr_match = off["correct"] == vlm["correct"]
        exact_text += int(text_match)
        same_pred += int(pred_match)
        same_correct += int(corr_match)
        paired.append(
            {
                "i": idx,
                "gold": off["gold"],
                "offline_predicted": off["predicted"],
                "vllm_predicted": vlm["predicted"],
                "offline_correct": off["correct"],
                "vllm_correct": vlm["correct"],
                "text_exact_match": text_match,
                "prediction_match": pred_match,
                "correctness_match": corr_match,
                "offline_text": off["text"],
                "vllm_text": vlm["text"],
            }
        )
    n = len(ids)
    result = {
        "n": n,
        "offline_accuracy": offline["accuracy"],
        "vllm_accuracy": vllm["accuracy"],
        "accuracy_delta_vllm_minus_offline": vllm["accuracy"] - offline["accuracy"],
        "exact_text_match_rate": exact_text / n,
        "prediction_match_rate": same_pred / n,
        "correctness_match_rate": same_correct / n,
        "rows": paired,
    }
    out = Path(os.path.expanduser(out_path))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    print(f"comparison written={out}")


def run_all(
    limit: int = 20,
    max_tokens: int = 64,
    receiver_model: str = "Qwen/Qwen3-0.6B",
    producer_model: str = "Qwen/Qwen3-4B-Base",
    fuser_dir: str = "",
    out_dir: str = "~/bench_logs/c2c_arc_equivalence",
) -> None:
    root = Path(os.path.expanduser(out_dir))
    offline_path = root / "offline.json"
    vllm_path = root / "vllm.json"
    comparison_path = root / "comparison.json"
    run_offline(
        limit=limit,
        max_tokens=max_tokens,
        receiver_model=receiver_model,
        producer_model=producer_model,
        fuser_dir=fuser_dir,
        out_path=str(offline_path),
    )
    run_vllm(
        limit=limit,
        max_tokens=max_tokens,
        receiver_model=receiver_model,
        producer_model=producer_model,
        fuser_dir=fuser_dir,
        out_path=str(vllm_path),
        log_dir=str(root / "servers"),
    )
    compare(
        offline_path=str(offline_path),
        vllm_path=str(vllm_path),
        out_path=str(comparison_path),
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "run_offline": run_offline,
            "run_vllm": run_vllm,
            "compare": compare,
            "run_all": run_all,
        }
    )
