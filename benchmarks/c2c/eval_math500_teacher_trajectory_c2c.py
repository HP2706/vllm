"""Evaluate C2C conditioning on teacher-generated MATH500 trajectories.

For each MATH500 problem:
1. Generate a full teacher trajectory with Qwen3-4B-Base.
2. Split that generated trajectory into evenly spaced prefix cutpoints.
3. For each prefix, prefill the teacher/producer on the exact prefix token IDs,
   fuse that cache into the weaker Qwen3-0.6B receiver, and let the receiver
   continue.
4. Optionally compare against a text-only Qwen3-0.6B receiver on the same prefix.

The C2C path uses TCP transport and the Rosetta-style deferred fusion mode, so
the receiver first writes clean prompt KV and then applies all projector writes
before the final continuation request prefix-hits the fused cache.
"""

import json
import math
import os
import re
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import fire
import requests
from datasets import load_dataset
from transformers import AutoTokenizer


@dataclass(frozen=True)
class MathProblem:
    index: int
    problem: str
    answer: str
    subject: str
    level: int
    unique_id: str


def _load_math500(limit: int) -> list[MathProblem]:
    dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
    n = min(limit, len(dataset))
    problems: list[MathProblem] = []
    for i in range(n):
        row = dataset[i]
        problems.append(
            MathProblem(
                index=i,
                problem=row["problem"],
                answer=row["answer"],
                subject=row["subject"],
                level=int(row["level"]),
                unique_id=row["unique_id"],
            )
        )
    return problems


def _math_prompt(problem: str) -> str:
    return (
        "Solve the following math problem. Think step by step, and put the final "
        f"answer in \\boxed{{}}.\n\nProblem:\n{problem}"
    )


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


def _completion(
    port: int,
    model: str,
    prompt: str | list[int],
    max_tokens: int,
    temperature: float,
    timeout_s: float = 600.0,
) -> tuple[str, float]:
    t0 = time.perf_counter()
    response = requests.post(
        f"http://localhost:{port}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=timeout_s,
    )
    response.raise_for_status()
    text = response.json()["choices"][0]["text"]
    return text, (time.perf_counter() - t0) * 1000.0


def _chat_prompt_ids(tokenizer: Any, problem: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": _math_prompt(problem)}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def _trajectory_cutpoints(num_tokens: int, num_parts: int) -> list[int]:
    if num_tokens <= 0:
        return [0 for _ in range(num_parts)]
    return [
        min(num_tokens, max(1, math.ceil(num_tokens * part / num_parts)))
        for part in range(1, num_parts + 1)
    ]


def _extract_boxed(text: str) -> str | None:
    starts = [match.start() for match in re.finditer(r"\\boxed\{", text)]
    if not starts:
        return None
    start = starts[-1] + len(r"\boxed{")
    depth = 1
    chars: list[str] = []
    for ch in text[start:]:
        if ch == "{":
            depth += 1
            chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars)
            chars.append(ch)
        else:
            chars.append(ch)
    return None


def _extract_answer(text: str) -> str:
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed
    answer_match = re.search(r"(?:final answer|answer)\s*[:=]\s*(.+)", text, re.I | re.S)
    if answer_match is not None:
        return answer_match.group(1).strip().splitlines()[0]
    return text.strip().splitlines()[-1] if text.strip() else ""


def _normalize_answer(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.strip("$")
    cleaned = cleaned.replace("\\left", "").replace("\\right", "")
    cleaned = cleaned.replace("\\,", "").replace("\\!", "")
    cleaned = cleaned.replace(" ", "").replace("\n", "")
    cleaned = cleaned.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return cleaned.rstrip(".")


def _judge(text: str, answer: str) -> dict[str, Any]:
    extracted = _extract_answer(text)
    ok = _normalize_answer(extracted) == _normalize_answer(answer)
    return {
        "correct": bool(ok),
        "extracted": str(extracted),
        "judge": "boxed_or_answer_exact_normalized",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def run(
    limit: int = 10,
    num_parts: int = 10,
    teacher_model: str = "Qwen/Qwen3-4B-Base",
    student_model: str = "Qwen/Qwen3-0.6B",
    fuser_dir: str = "",
    out_dir: str = "~/bench_logs/math500_teacher_trajectory_c2c",
    teacher_max_tokens: int = 1024,
    student_max_tokens: int = 512,
    teacher_temperature: float = 0.6,
    student_temperature: float = 0.0,
    max_model_len: int = 4096,
    producer_gpu: int = 0,
    consumer_gpu: int = 1,
    producer_port: int = 8100,
    c2c_student_port: int = 8200,
    text_student_port: int = 8300,
    c2c_port: int = 8077,
    include_text_baseline: bool = False,
) -> None:
    assert fuser_dir, "Pass --fuser-dir pointing at the C2C fuser final directory"
    root = Path(os.path.expanduser(out_dir))
    root.mkdir(parents=True, exist_ok=True)
    timing_dir = root / "timings"
    timing_dir.mkdir(parents=True, exist_ok=True)
    logs = root / "servers"
    logs.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(student_model)
    problems = _load_math500(limit)

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
        "c2c_fusion_timing": "deferred",
    }
    common = [
        "--max-model-len",
        str(max_model_len),
        "--max-num-batched-tokens",
        str(max_model_len),
        "--enforce-eager",
    ]
    prod_env = os.environ.copy()
    prod_env["CUDA_VISIBLE_DEVICES"] = str(producer_gpu)
    cons_env = os.environ.copy()
    cons_env["CUDA_VISIBLE_DEVICES"] = str(consumer_gpu)

    processes: list[subprocess.Popen[bytes]] = []
    producer_log = open(logs / "producer.txt", "wb")
    c2c_log = open(logs / "student_c2c.txt", "wb")
    text_log = open(logs / "student_text.txt", "wb")
    result_path = root / "results.json"
    started_at = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    result: dict[str, Any] = {
        "started_at": started_at,
        "dataset": "HuggingFaceH4/MATH-500/test",
        "limit": limit,
        "num_parts": num_parts,
        "teacher_model": teacher_model,
        "student_model": student_model,
        "fuser_dir": fuser_dir,
        "teacher_max_tokens": teacher_max_tokens,
        "student_max_tokens": student_max_tokens,
        "max_model_len": max_model_len,
        "include_text_baseline": include_text_baseline,
        "rows": [],
    }
    try:
        processes.append(
            subprocess.Popen(
                [
                    "vllm",
                    "serve",
                    teacher_model,
                    "--port",
                    str(producer_port),
                    "--kv-transfer-config",
                    _kv_config("kv_producer", producer_extra),
                    "--gpu-memory-utilization",
                    "0.45",
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
                    student_model,
                    "--port",
                    str(c2c_student_port),
                    "--kv-transfer-config",
                    _kv_config("kv_consumer", consumer_extra),
                    "--gpu-memory-utilization",
                    "0.25",
                    *common,
                ],
                env=cons_env,
                stdout=c2c_log,
                stderr=subprocess.STDOUT,
            )
        )
        if include_text_baseline:
            processes.append(
                subprocess.Popen(
                    [
                        "vllm",
                        "serve",
                        student_model,
                        "--port",
                        str(text_student_port),
                        "--gpu-memory-utilization",
                        "0.25",
                        *common,
                    ],
                    env=cons_env,
                    stdout=text_log,
                    stderr=subprocess.STDOUT,
                )
            )
        _wait_port(producer_port)
        _wait_port(c2c_student_port)
        if include_text_baseline:
            _wait_port(text_student_port)

        rows: list[dict[str, Any]] = []
        for problem in problems:
            base_ids = _chat_prompt_ids(tokenizer, problem.problem)
            teacher_text, teacher_ms = _completion(
                producer_port,
                teacher_model,
                base_ids,
                teacher_max_tokens,
                teacher_temperature,
            )
            teacher_judge = _judge(teacher_text, problem.answer)
            teacher_tokens = tokenizer(
                teacher_text,
                add_special_tokens=False,
            )["input_ids"]
            cutpoints = _trajectory_cutpoints(len(teacher_tokens), num_parts)

            student_original: dict[str, Any] | None = None
            if include_text_baseline:
                student_text, student_ms = _completion(
                    text_student_port,
                    student_model,
                    base_ids,
                    student_max_tokens,
                    student_temperature,
                )
                student_original = {
                    **_judge(student_text, problem.answer),
                    "continuation": student_text,
                    "latency_ms": student_ms,
                }

            prefix_rows: list[dict[str, Any]] = []
            for part_idx, cut in enumerate(cutpoints, start=1):
                prefix_ids = base_ids + teacher_tokens[:cut]
                prefix_text = tokenizer.decode(
                    teacher_tokens[:cut],
                    skip_special_tokens=False,
                )
                text_only: dict[str, Any] | None = None
                if include_text_baseline:
                    text_continuation, text_ms = _completion(
                        text_student_port,
                        student_model,
                        prefix_ids,
                        student_max_tokens,
                        student_temperature,
                    )
                    text_full = prefix_text + text_continuation
                    text_only = {
                        **_judge(text_full, problem.answer),
                        "continuation": text_continuation,
                        "latency_ms": text_ms,
                    }
                _, producer_ms = _completion(
                    producer_port,
                    teacher_model,
                    prefix_ids,
                    1,
                    0.0,
                )
                _, fusion_ms = _completion(
                    c2c_student_port,
                    student_model,
                    prefix_ids,
                    1,
                    0.0,
                )
                c2c_continuation, c2c_ms = _completion(
                    c2c_student_port,
                    student_model,
                    prefix_ids,
                    student_max_tokens,
                    student_temperature,
                )
                c2c_full = prefix_text + c2c_continuation
                prefix_rows.append(
                    {
                        "part": part_idx,
                        "trajectory_token_cut": cut,
                        "trajectory_token_total": len(teacher_tokens),
                        "text_only": text_only,
                        "c2c": {
                            **_judge(c2c_full, problem.answer),
                            "continuation": c2c_continuation,
                            "producer_prefill_ms": producer_ms,
                            "fusion_prefill_ms": fusion_ms,
                            "continuation_ms": c2c_ms,
                        },
                    }
                )

            row = {
                "index": problem.index,
                "unique_id": problem.unique_id,
                "subject": problem.subject,
                "level": problem.level,
                "answer": problem.answer,
                "problem": problem.problem,
                "teacher": {
                    **teacher_judge,
                    "trajectory": teacher_text,
                    "trajectory_tokens": len(teacher_tokens),
                    "latency_ms": teacher_ms,
                },
                "student_original": student_original,
                "prefixes": prefix_rows,
            }
            rows.append(row)
            result["rows"] = rows
            _write_json(result_path, result)
            print(
                json.dumps(
                    {
                        "index": problem.index,
                        "teacher_correct": teacher_judge["correct"],
                        "student_original_correct": None
                        if student_original is None
                        else student_original["correct"],
                        "c2c_correct_by_part": [
                            prefix["c2c"]["correct"] for prefix in prefix_rows
                        ],
                        "text_correct_by_part": None
                        if not include_text_baseline
                        else [prefix["text_only"]["correct"] for prefix in prefix_rows],
                    }
                ),
                flush=True,
            )
    finally:
        _terminate(processes)
        producer_log.close()
        c2c_log.close()
        text_log.close()

    rows = result["rows"]
    teacher_correct = sum(int(row["teacher"]["correct"]) for row in rows)
    student_correct = (
        None
        if not include_text_baseline
        else sum(int(row["student_original"]["correct"]) for row in rows)
    )
    by_part: list[dict[str, Any]] = []
    for part_idx in range(1, num_parts + 1):
        text_correct = 0
        c2c_correct = 0
        for row in rows:
            prefix = row["prefixes"][part_idx - 1]
            if include_text_baseline:
                text_correct += int(prefix["text_only"]["correct"])
            c2c_correct += int(prefix["c2c"]["correct"])
        by_part.append(
            {
                "part": part_idx,
                "text_only_accuracy": None
                if not include_text_baseline
                else text_correct / len(rows),
                "c2c_accuracy": c2c_correct / len(rows),
                "delta_c2c_minus_text": None
                if not include_text_baseline
                else (c2c_correct - text_correct) / len(rows),
            }
        )
    result["summary"] = {
        "n": len(rows),
        "teacher_accuracy": teacher_correct / len(rows),
        "student_original_accuracy": None
        if student_correct is None
        else student_correct / len(rows),
        "by_part": by_part,
    }
    _write_json(result_path, result)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    fire.Fire({"run": run})
