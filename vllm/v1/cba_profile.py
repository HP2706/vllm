# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in JSONL profiler for cross-batch attention investigation."""

import json
import os
import time
from contextlib import contextmanager
from typing import Any

import torch


def enabled() -> bool:
    return bool(os.environ.get("VLLM_CBA_PROFILE_LOG"))


def log_event(event: str, **fields: Any) -> None:
    path = os.environ.get("VLLM_CBA_PROFILE_LOG")
    if not path:
        return
    payload = {
        "ts": time.time(),
        "pid": os.getpid(),
        "event": event,
        **fields,
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass


@contextmanager
def cpu_timer(event: str, **fields: Any):
    if not enabled():
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        log_event(event, elapsed_ms=(time.perf_counter() - start) * 1000.0, **fields)


@contextmanager
def cuda_timer(event: str, **fields: Any):
    if not enabled():
        yield
        return
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        log_event(event, elapsed_ms=(time.perf_counter() - start) * 1000.0, **fields)
