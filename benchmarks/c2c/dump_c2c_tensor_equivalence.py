"""Dump and compare C2C cache-space tensors for Rosetta vs native vLLM.

This is a tensor-level equivalence harness for the C2C implementation.  The
reference side uses the public thu-nics/C2C Rosetta model and saves the tensors
around the exact projector contract:

* source model post-RoPE KV slice
* receiver model post-RoPE KV slice before fusion
* receiver KV slice after the trained projector

The vLLM side enables the same dump inside ``C2CConnector._run_fusion_layer``.
The comparator then checks that transport, token indexing, layer mapping, and
projector application produce the same cache-space tensors.
"""

import gc
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fire
import requests
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

import rosetta.model.wrapper as rosetta_wrapper
from rosetta.utils.evaluate import build_prompt, load_rosetta_model


@dataclass(frozen=True)
class ArcPrompt:
    index: int
    prompt: str
    gold: str


def _prompt_key(token_ids: list[int]) -> str:
    h = hashlib.sha256(b",".join(str(t).encode() for t in token_ids))
    return h.hexdigest()[:24]


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


def _load_prompt(index: int) -> ArcPrompt:
    dataset = load_dataset("ai2_arc", "ARC-Challenge")["test"]
    return _arc_prompt(dataset[index], index)


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
                "max_new_tokens": 1,
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


def _tokenize_prompt(tokenizer: Any, prompt: str, device: torch.device) -> dict[str, torch.Tensor]:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(text, return_tensors="pt").to(device)


def _kv_cache_index(instruction_length: int, device: torch.device) -> list[torch.Tensor]:
    instruction_index = (
        torch.tensor([1, 0], dtype=torch.long, device=device)
        .repeat(instruction_length, 1)
        .unsqueeze(0)
    )
    response_index = (
        torch.tensor([-1, 0], dtype=torch.long, device=device)
        .repeat(1, 1)
        .unsqueeze(0)
    )
    return [instruction_index, response_index]


def _parse_layers(raw: str | tuple[int, ...] | list[int] | int) -> set[int]:
    if isinstance(raw, int):
        return {raw}
    if isinstance(raw, tuple) or isinstance(raw, list):
        return {int(part) for part in raw}
    if not raw:
        return set()
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def dump_offline(
    example_index: int = 0,
    receiver_model: str = "Qwen/Qwen3-0.6B",
    producer_model: str = "Qwen/Qwen3-4B-Base",
    fuser_dir: str = "",
    out_dir: str = "~/bench_logs/c2c_tensor_equivalence/offline",
    layers: str = "0,1,2,27",
    token_limit: int = 8,
) -> None:
    assert fuser_dir, "Pass --fuser-dir pointing at the C2C fuser final directory"
    out = Path(os.path.expanduser(out_dir))
    out.mkdir(parents=True, exist_ok=True)
    selected_layers = _parse_layers(layers)
    device = torch.device("cuda:0")
    prompt = _load_prompt(example_index)
    config = _rosetta_config(
        receiver_model=receiver_model,
        producer_model=producer_model,
        fuser_dir=fuser_dir,
        out_dir=str(out),
    )
    model, tokenizer = load_rosetta_model(config["model"], config["eval"], device)
    model.eval()
    tokenized = _tokenize_prompt(tokenizer, prompt.prompt, device)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    position_ids = attention_mask.long().cumsum(-1) - 1
    instruction_length = input_ids.shape[1] - 1
    token_ids = input_ids[0].detach().cpu().tolist()
    key = _prompt_key(token_ids)
    with torch.inference_mode():
        output = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache_index=_kv_cache_index(instruction_length, device),
            use_cache=True,
        )
    base_cache = model.kv_cache_dict[model.base_model_idx][model.base_model_idx]
    source_model_idx = 1
    source_cache = model.kv_cache_dict[model.base_model_idx][source_model_idx]
    layer_map = model.projector_dict[model.base_model_idx][source_model_idx]
    n_dump = min(token_limit, instruction_length)
    positions = torch.arange(0, n_dump, dtype=torch.long)
    written: list[str] = []
    for target_layer_idx, entry in layer_map.items():
        if selected_layers and target_layer_idx not in selected_layers:
            continue
        source_layer_idx, projector_idx = entry[0]
        base_key_cache, base_value_cache = base_cache[target_layer_idx]
        source_key_cache, source_value_cache = source_cache[source_layer_idx]
        target_key = base_key_cache[:, :, :instruction_length, :]
        target_value = base_value_cache[:, :, :instruction_length, :]
        source_key = source_key_cache[:, :, :instruction_length, :]
        source_value = source_value_cache[:, :, :instruction_length, :]
        projector = model.projector_list[projector_idx]
        fused_key, fused_value = projector(
            (source_key, source_value),
            (target_key, target_value),
        )
        dump = {
            "path": "offline_rosetta",
            "key": key,
            "example_index": example_index,
            "target_layer": target_layer_idx,
            "source_layer": source_layer_idx,
            "projector_idx": projector_idx,
            "positions": positions,
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "source_key": source_key[:, :, :n_dump, :].detach().float().cpu(),
            "source_value": source_value[:, :, :n_dump, :].detach().float().cpu(),
            "target_key_before": target_key[:, :, :n_dump, :].detach().float().cpu(),
            "target_value_before": target_value[:, :, :n_dump, :].detach().float().cpu(),
            "fused_key": fused_key[:, :, :n_dump, :].detach().float().cpu(),
            "fused_value": fused_value[:, :, :n_dump, :].detach().float().cpu(),
            "next_token_logits": output.logits[:, -1, :].detach().float().cpu(),
        }
        path = out / f"{key}_layer_{target_layer_idx}.pt"
        torch.save(dump, path)
        written.append(str(path))
    meta = {
        "path": "offline_rosetta",
        "key": key,
        "example_index": example_index,
        "receiver_model": receiver_model,
        "producer_model": producer_model,
        "fuser_dir": fuser_dir,
        "prompt_tokens": len(token_ids),
        "instruction_length": instruction_length,
        "layers": sorted(selected_layers),
        "token_limit": token_limit,
        "files": written,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


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


def _completion(port: int, model: str, prompt: list[int], max_tokens: int) -> str:
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
    return response.json()["choices"][0]["text"]


def dump_vllm(
    example_index: int = 0,
    receiver_model: str = "Qwen/Qwen3-0.6B",
    producer_model: str = "Qwen/Qwen3-4B-Base",
    fuser_dir: str = "",
    out_dir: str = "~/bench_logs/c2c_tensor_equivalence/vllm",
    log_dir: str = "~/bench_logs/c2c_tensor_equivalence/servers",
    layers: str = "0,1,2,27",
    token_limit: int = 8,
    producer_gpu: int = 0,
    consumer_gpu: int = 1,
    producer_port: int = 8100,
    consumer_port: int = 8200,
    c2c_port: int = 8077,
) -> None:
    assert fuser_dir, "Pass --fuser-dir pointing at the C2C fuser final directory"
    out = Path(os.path.expanduser(out_dir))
    logs = Path(os.path.expanduser(log_dir))
    timing_dir = logs / "timings"
    out.mkdir(parents=True, exist_ok=True)
    timing_dir.mkdir(parents=True, exist_ok=True)
    prompt = _load_prompt(example_index)
    tokenizer = AutoTokenizer.from_pretrained(receiver_model)
    prompt_ids: list[int] = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    key = _prompt_key(prompt_ids)
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
        "c2c_tensor_dump_dir": str(out),
        "c2c_tensor_dump_layers": layers,
        "c2c_tensor_dump_token_limit": token_limit,
        "c2c_fusion_timing": "deferred",
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
    producer_log = open(logs / "producer.txt", "wb")
    consumer_log = open(logs / "consumer.txt", "wb")
    processes: list[subprocess.Popen[bytes]] = []
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
        _completion(producer_port, producer_model, prompt_ids, 1)
        _completion(consumer_port, receiver_model, prompt_ids, 1)
    finally:
        _terminate(processes)
        producer_log.close()
        consumer_log.close()
    files = sorted(str(path) for path in out.glob(f"{key}_layer_*.pt"))
    meta = {
        "path": "vllm",
        "key": key,
        "example_index": example_index,
        "receiver_model": receiver_model,
        "producer_model": producer_model,
        "fuser_dir": fuser_dir,
        "prompt_tokens": len(prompt_ids),
        "layers": sorted(_parse_layers(layers)),
        "token_limit": token_limit,
        "files": files,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def _tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    assert a.shape == b.shape, f"shape mismatch: {tuple(a.shape)} != {tuple(b.shape)}"
    a32 = a.float()
    b32 = b.float()
    diff = (a32 - b32).abs()
    flat_a = a32.reshape(-1)
    flat_b = b32.reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(flat_a, flat_b, dim=0)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rms": float(torch.sqrt(torch.mean((a32 - b32) ** 2)).item()),
        "cosine": float(cosine.item()),
    }


def compare(
    offline_dir: str = "~/bench_logs/c2c_tensor_equivalence/offline",
    vllm_dir: str = "~/bench_logs/c2c_tensor_equivalence/vllm",
    out_path: str = "~/bench_logs/c2c_tensor_equivalence/comparison.json",
) -> None:
    offline_root = Path(os.path.expanduser(offline_dir))
    vllm_root = Path(os.path.expanduser(vllm_dir))
    offline_meta = json.loads((offline_root / "meta.json").read_text())
    vllm_meta = json.loads((vllm_root / "meta.json").read_text())
    assert offline_meta["key"] == vllm_meta["key"], "prompt key mismatch"
    key = offline_meta["key"]
    fields = [
        "source_key",
        "source_value",
        "target_key_before",
        "target_value_before",
        "fused_key",
        "fused_value",
    ]
    rows: list[dict[str, Any]] = []
    for offline_path in sorted(offline_root.glob(f"{key}_layer_*.pt")):
        layer = int(offline_path.stem.rsplit("_", 1)[1])
        vllm_path = vllm_root / offline_path.name
        assert vllm_path.exists(), f"missing vLLM dump for layer {layer}: {vllm_path}"
        offline = torch.load(offline_path, map_location="cpu")
        vllm = torch.load(vllm_path, map_location="cpu")
        assert int(offline["target_layer"]) == int(vllm["target_layer"])
        assert int(offline["source_layer"]) == int(vllm["source_layer"])
        field_metrics = {
            field: _tensor_metrics(offline[field], vllm[field]) for field in fields
        }
        rows.append(
            {
                "target_layer": layer,
                "source_layer": int(offline["source_layer"]),
                "projector_idx": int(offline["projector_idx"]),
                "positions_match": bool(torch.equal(offline["positions"], vllm["positions"])),
                "fields": field_metrics,
            }
        )
    max_abs = max(
        metric["max_abs"]
        for row in rows
        for metric in row["fields"].values()
    )
    min_cosine = min(
        metric["cosine"]
        for row in rows
        for metric in row["fields"].values()
    )
    result = {
        "key": key,
        "offline_dir": str(offline_root),
        "vllm_dir": str(vllm_root),
        "num_layers": len(rows),
        "global_max_abs": max_abs,
        "global_min_cosine": min_cosine,
        "rows": rows,
    }
    out = Path(os.path.expanduser(out_path))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def run_all(
    example_index: int = 0,
    receiver_model: str = "Qwen/Qwen3-0.6B",
    producer_model: str = "Qwen/Qwen3-4B-Base",
    fuser_dir: str = "",
    out_dir: str = "~/bench_logs/c2c_tensor_equivalence",
    layers: str = "0,1,2,27",
    token_limit: int = 8,
) -> None:
    root = Path(os.path.expanduser(out_dir))
    dump_offline(
        example_index=example_index,
        receiver_model=receiver_model,
        producer_model=producer_model,
        fuser_dir=fuser_dir,
        out_dir=str(root / "offline"),
        layers=layers,
        token_limit=token_limit,
    )
    dump_vllm(
        example_index=example_index,
        receiver_model=receiver_model,
        producer_model=producer_model,
        fuser_dir=fuser_dir,
        out_dir=str(root / "vllm"),
        log_dir=str(root / "servers"),
        layers=layers,
        token_limit=token_limit,
    )
    compare(
        offline_dir=str(root / "offline"),
        vllm_dir=str(root / "vllm"),
        out_path=str(root / "comparison.json"),
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "dump_offline": dump_offline,
            "dump_vllm": dump_vllm,
            "compare": compare,
            "run_all": run_all,
        }
    )
