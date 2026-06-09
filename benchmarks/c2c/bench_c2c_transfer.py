"""Benchmark C2C-style KV cache transfer between two vLLM servers on one GPU.

Spawns:
  - producer: Qwen/Qwen2.5-0.5B-Instruct (sharer) with C2CConnector kv_producer
  - consumer: Qwen/Qwen3-0.6B (receiver) with C2CConnector kv_consumer

For each prompt length, measures:
  - producer prefill latency (API wall time, max_tokens=1)
  - consumer prefill latency with KV transfer happening inside the forward
  - in-connector timings: IPC open ms, device-to-device copy ms, bandwidth
  - text-handoff baseline: producer generates a 64-token summary, consumer
    prefills prompt+summary (what C2C aims to beat)

Run inside the vllm venv on the pod:
  python bench_c2c_transfer.py run
"""

import json
import os
import shutil
import subprocess
import time
from typing import Any

import fire
import requests
from transformers import AutoTokenizer

C2C_DIR = "/dev/shm/c2c_kv"
PRODUCER_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CONSUMER_MODEL = "Qwen/Qwen3-0.6B"
PRODUCER_PORT = 8100
CONSUMER_PORT = 8200
LOG_DIR = os.path.expanduser("~/bench_logs")

BASE_SENTENCE = (
    "The history of thermodynamics begins with the study of heat engines, "
    "where early scientists sought to understand how thermal energy could be "
    "converted into mechanical work with maximum efficiency. "
)


def _server_cmd(model: str, port: int, kv_role: str) -> list[str]:
    kv_cfg = json.dumps(
        {
            "kv_connector": "C2CConnector",
            "kv_role": kv_role,
            "kv_connector_extra_config": {"c2c_dir": C2C_DIR},
        }
    )
    return [
        "vllm",
        "serve",
        model,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        "0.22",
        "--max-model-len",
        "16384",
        "--max-num-batched-tokens",
        "16384",
        "--no-enable-prefix-caching",
        "--enforce-eager",
        "--kv-transfer-config",
        kv_cfg,
    ]


def _wait_ready(port: int, timeout_s: float = 600.0) -> None:
    deadline = time.time() + timeout_s
    url = f"http://localhost:{port}/health"
    while time.time() < deadline:
        if os.system(f"curl -s -o /dev/null -w '' {url}") == 0:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return
        time.sleep(3)
    raise TimeoutError(f"server on port {port} not ready after {timeout_s}s")


def _completion(
    port: int, prompt: str, max_tokens: int, model: str
) -> tuple[float, str]:
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
    dt = time.perf_counter() - t0
    r.raise_for_status()
    return dt * 1000.0, r.json()["choices"][0]["text"]


def _build_prompt(tokenizer: Any, target_tokens: int) -> str:
    text = ""
    while len(tokenizer(text)["input_ids"]) < target_tokens:
        text += BASE_SENTENCE
    ids = tokenizer(text)["input_ids"][:target_tokens]
    return tokenizer.decode(ids)


def run(prompt_lengths: tuple[int, ...] = (256, 1024, 4096, 8192)) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    shutil.rmtree(C2C_DIR, ignore_errors=True)
    os.makedirs(C2C_DIR, exist_ok=True)

    procs: list[subprocess.Popen[bytes]] = []
    for model, port, role in [
        (PRODUCER_MODEL, PRODUCER_PORT, "kv_producer"),
        (CONSUMER_MODEL, CONSUMER_PORT, "kv_consumer"),
    ]:
        log = open(os.path.join(LOG_DIR, f"server_{port}.txt"), "w")
        procs.append(
            subprocess.Popen(
                _server_cmd(model, port, role), stdout=log, stderr=subprocess.STDOUT
            )
        )
    print("waiting for servers...")
    _wait_ready(PRODUCER_PORT)
    _wait_ready(CONSUMER_PORT)
    print("servers ready")

    tok = AutoTokenizer.from_pretrained(CONSUMER_MODEL)
    results: list[dict[str, Any]] = []
    for n in prompt_lengths:
        prompt = _build_prompt(tok, n)
        n_actual = len(tok(prompt)["input_ids"])

        # 1) consumer prefill WITHOUT transfer (no manifest exists yet)
        consumer_solo_ms, _ = _completion(CONSUMER_PORT, prompt, 1, CONSUMER_MODEL)

        # 2) producer prefill (stages KV + publishes IPC handle)
        producer_ms, _ = _completion(PRODUCER_PORT, prompt, 1, PRODUCER_MODEL)
        time.sleep(0.5)

        # 3) consumer prefill WITH transfer (same prompt text, new request).
        # prefix caching is off, so the consumer recomputes prefill while the
        # connector pulls the producer KV inside the same forward.
        consumer_xfer_ms, _ = _completion(CONSUMER_PORT, prompt, 1, CONSUMER_MODEL)

        # in-connector timing
        key_timings: list[dict[str, Any]] = []
        for fname in os.listdir(C2C_DIR):
            if fname.startswith("timing_"):
                with open(os.path.join(C2C_DIR, fname)) as f:
                    key_timings.append(json.load(f))
        key_timings.sort(key=lambda d: d.get("num_prompt_tokens_consumer", 0))
        latest = key_timings[-1] if key_timings else {}

        # 4) text handoff baseline: producer writes a summary, consumer
        # prefills prompt + summary
        handoff_gen_ms, summary = _completion(PRODUCER_PORT, prompt, 64, PRODUCER_MODEL)
        handoff_prefill_ms, _ = _completion(
            CONSUMER_PORT, prompt + "\n" + summary, 1, CONSUMER_MODEL
        )

        row = {
            "prompt_tokens": n_actual,
            "producer_prefill_ms": round(producer_ms, 1),
            "consumer_prefill_solo_ms": round(consumer_solo_ms, 1),
            "consumer_prefill_with_transfer_ms": round(consumer_xfer_ms, 1),
            "ipc_open_ms": round(latest.get("manifest_open_ms", -1), 3),
            "ipc_copy_ms": round(latest.get("ipc_copy_ms", -1), 3),
            "transfer_bandwidth_gb_s": round(latest.get("bandwidth_gb_s") or -1, 1),
            "transfer_bytes_mb": round(latest.get("bytes", 0) / 1e6, 2),
            "producer_extract_ms": round(latest.get("producer_extract_ms", -1), 3),
            "text_handoff_gen_ms": round(handoff_gen_ms, 1),
            "text_handoff_prefill_ms": round(handoff_prefill_ms, 1),
        }
        results.append(row)
        print(json.dumps(row))
        for fname in os.listdir(C2C_DIR):
            os.remove(os.path.join(C2C_DIR, fname))

    out = os.path.join(LOG_DIR, "c2c_transfer_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults written to {out}")

    cols = list(results[0].keys())
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for r in results:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")

    for p in procs:
        p.terminate()
    for p in procs:
        p.wait(timeout=60)
    print("servers stopped")


if __name__ == "__main__":
    fire.Fire({"run": run})
