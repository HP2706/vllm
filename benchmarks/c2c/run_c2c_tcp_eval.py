"""Launch a no-/dev/shm C2C vLLM eval.

This runner starts two vLLM servers with ``C2CConnector`` configured for
``c2c_transport=tcp``:

* producer: Qwen/Qwen2.5-0.5B-Instruct on GPU 0
* consumer: Qwen/Qwen3-0.6B on GPU 1 with the trained C2C fuser

The eval still uses ``eval_c2c_vllm.py`` for prompt formatting and scoring, but
the KV handoff goes through the connector's TCP transport and the receiver-side
``mutate_kv_post_write`` hook. No shared-memory manifest directory is used.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import fire


PRODUCER_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CONSUMER_MODEL = "Qwen/Qwen3-0.6B"


def _kv_config(role: str, extra: dict[str, Any]) -> str:
    return json.dumps(
        {
            "kv_connector": "C2CConnector",
            "kv_role": role,
            "kv_connector_extra_config": extra,
        }
    )


def _wait_health(port: int, timeout_s: float = 300.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2.0)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(3)
    raise TimeoutError(f"vLLM server on port {port} did not become healthy")


def _terminate(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
    for process in processes:
        if process.poll() is None:
            process.wait(timeout=30)


def run(
    limit: int = 100,
    max_tokens: int = 64,
    producer_gpu: int = 0,
    consumer_gpu: int = 1,
    producer_port: int = 8100,
    consumer_port: int = 8200,
    c2c_port: int = 8077,
    fuser_dir: str = "",
    out_dir: str = "~/bench_logs",
) -> None:
    assert fuser_dir, "Pass --fuser-dir pointing at the C2C fuser final directory"
    log_dir = Path(os.path.expanduser(out_dir))
    timing_dir = log_dir / "c2c_tcp_timings"
    log_dir.mkdir(parents=True, exist_ok=True)
    timing_dir.mkdir(parents=True, exist_ok=True)

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

    common_args = [
        "--gpu-memory-utilization",
        "0.22",
        "--max-model-len",
        "16384",
        "--max-num-batched-tokens",
        "16384",
        "--enforce-eager",
    ]
    producer_env = os.environ.copy()
    producer_env["CUDA_VISIBLE_DEVICES"] = str(producer_gpu)
    consumer_env = os.environ.copy()
    consumer_env["CUDA_VISIBLE_DEVICES"] = str(consumer_gpu)

    producer_log = open(log_dir / "server_c2c_tcp_producer.txt", "wb")
    consumer_log = open(log_dir / "server_c2c_tcp_consumer.txt", "wb")
    processes: list[subprocess.Popen[bytes]] = []
    try:
        processes.append(
            subprocess.Popen(
                [
                    "vllm",
                    "serve",
                    PRODUCER_MODEL,
                    "--port",
                    str(producer_port),
                    "--kv-transfer-config",
                    _kv_config("kv_producer", producer_extra),
                    *common_args,
                ],
                env=producer_env,
                stdout=producer_log,
                stderr=subprocess.STDOUT,
            )
        )
        processes.append(
            subprocess.Popen(
                [
                    "vllm",
                    "serve",
                    CONSUMER_MODEL,
                    "--port",
                    str(consumer_port),
                    "--kv-transfer-config",
                    _kv_config("kv_consumer", consumer_extra),
                    *common_args,
                ],
                env=consumer_env,
                stdout=consumer_log,
                stderr=subprocess.STDOUT,
            )
        )
        _wait_health(producer_port)
        _wait_health(consumer_port)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("eval_c2c_vllm.py")),
                "run",
                "--mode",
                "fused",
                "--limit",
                str(limit),
                "--max_tokens",
                str(max_tokens),
            ],
            check=True,
        )
    finally:
        _terminate(processes)
        producer_log.close()
        consumer_log.close()


if __name__ == "__main__":
    fire.Fire({"run": run})
