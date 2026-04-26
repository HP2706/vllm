# vLLM Cross-Batch Attention Implementation State

Branch: `feature/cross-batch-flex-attention`

Base implementation commit: `f32aa74b7` (`Add V1 cross-batch FlexAttention prototype`)

Latest local progress:

- `410f173b1` (`Harden cross-batch FlexAttention prototype`)
- `97965e321` (`Wire cross-batch metadata through active GPU runner`)
- Current slice adds a Qwen3-0.6B HF-reference numerical assertion.

## What Is Implemented

- Added V1-local cross-batch attention metadata in
  `vllm/v1/cross_batch_attention.py`.
- Public research API is currently
  `SamplingParams.extra_args["cross_batch_attention"]` with:
  - `enabled`
  - `group_id`
  - `replica_id`
  - `group_size`
  - `virtual_token_id`
  - `virtual_window_size`
- `Request` parses and validates this metadata.
- `SchedulerOutput` carries request-id keyed cross-batch metadata for each
  scheduled step.
- The scheduler has conservative group preparation for waiting and running
  requests so enabled groups are kept together when the full group fits the
  current budget.
- Both V1 GPU runner paths translate request-id keyed group metadata into
  worker batch-order tensors after the existing request ordering.
- `CommonAttentionMetadata` and `FlexAttentionMetadata` carry optional
  cross-batch tensors.
- FlexAttention now supports a cross-batch decoder mask:
  - same-request causal attention remains allowed
  - peer attention is allowed only for same-group virtual tokens
  - peer virtual attention is causal in absolute logical position, matching
    `modeling_qwen3_batch_parscale.py`
- FlexAttention direct block-mask construction includes peer KV blocks for
  enabled same-group peers, then relies on the mask for exact token filtering.
- Cross-batch metadata now fails loudly if attention metadata is built for a
  non-FlexAttention backend.
- The scheduler now performs a conservative KV-capacity preflight before
  preparing a complete cross-batch group, so a group that cannot fit is skipped
  before any replica is admitted.
- Cross-batch batches force eager execution by constraining the batch descriptor
  to `CUDAGraphMode.NONE`.

## Tests Run

Environment:

- Local `.venv` under the vLLM repo
- `VLLM_USE_PRECOMPILED=1 python -m pip install -e .`
- `torch==2.10.0`
- GPU: `NVIDIA A100-SXM4-80GB`

Passing tests:

```bash
python -m pytest tests/v1/test_cross_batch_attention.py -q
```

Result after latest changes: `12 passed`

```bash
python -m pytest tests/kernels/test_flex_attention.py -q \
  -k physical_to_logical_mapping_handles_reused_blocks
```

Result: `1 passed, 3 deselected`

```bash
python -m pytest tests/v1/test_cross_batch_attention.py \
  tests/kernels/test_flex_attention.py -q \
  -k "cross_batch or physical_to_logical_mapping_handles_reused_blocks"
```

Result after latest changes: `15 passed, 3 deselected`

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_NO_USAGE_STATS=1 \
  python -m pytest tests/v1/test_cross_batch_attention_hf_reference.py -q -s
```

Result: `1 passed`

This loads `Qwen/Qwen3-0.6B` through both the HF-side
`src/modeling_qwen3_batch_parscale.py` reference and vLLM FlexAttention, then
asserts first-token selected logprobs are within `0.15` absolute tolerance and
that the HF top-5 token set is present in vLLM's returned top logprobs.

This includes:

- synthetic cross-batch direct-vs-slow FlexAttention block-mask coverage
- dense reference attention-output coverage showing non-virtual query outputs
  match baseline while virtual query outputs change with peer virtual context
- regression coverage for absolute-position virtual causality
- regression coverage that the active V1 `gpu_model_runner.py` builds
  cross-batch metadata in worker batch order, including padded rows
- rejection coverage for non-FlexAttention backends
- scheduler KV-capacity preflight coverage for grouped admission

```bash
git diff --check
python -m py_compile \
  vllm/v1/cross_batch_attention.py \
  vllm/v1/request.py \
  vllm/v1/core/sched/output.py \
  vllm/v1/core/sched/scheduler.py \
  vllm/v1/worker/gpu/input_batch.py \
  vllm/v1/worker/gpu/attn_utils.py \
  vllm/v1/worker/gpu/model_runner.py \
  vllm/v1/worker/gpu_model_runner.py \
  vllm/v1/worker/gpu/model_states/default.py \
  vllm/v1/attention/backend.py \
  vllm/v1/attention/backends/flex_attention.py \
  tests/kernels/test_flex_attention.py \
  tests/v1/test_cross_batch_attention.py
```

Result: passed.

Existing FlexAttention test attempted:

```bash
python -m pytest tests/kernels/test_flex_attention.py -q \
  -k "physical_to_logical_mapping_handles_reused_blocks or block_mask_direct_vs_slow_path"
```

Result: `physical_to_logical_mapping_handles_reused_blocks` passed, but
`test_block_mask_direct_vs_slow_path` failed because it tries to download gated
`meta-llama/Meta-Llama-3-8B` from Hugging Face and receives `401 Unauthorized`.
This failure is unrelated to the cross-batch implementation.

## Qwen3 Logit Probe

After wiring `vllm/v1/worker/gpu_model_runner.py`, the live Qwen3-0.6B
FlexAttention path now receives cross-batch metadata. A debug probe confirmed
that request 0's final prompt query at logical position 10 can see request 1's
virtual token at logical position 10 through the physical KV block mask.

Command shape:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_NO_USAGE_STATS=1 \
  .venv/bin/python <qwen3 baseline-vs-cross logprob script>
```

Configuration:

- model: `Qwen/Qwen3-0.6B`
- backend: `attention_config={"backend": "FLEX_ATTENTION"}`
- prompts:
  - `Solve briefly: 2 + 2 = reasoning`
  - `Solve briefly: 3 + 5 = reasoning`
- `virtual_token_id`: token id for `" reasoning"` (`32711`)
- `virtual_window_size`: `10`, aligning the HF reference position schedule with
  the final prompt token

Baseline top logprobs:

- request 0: `1882 -2.2030`, `3019 -2.3280`, `198 -2.5780`
- request 1: `3019 -1.9411`, `1882 -2.0661`, `198 -2.5661`

Cross-batch top logprobs:

- request 0: `1882 -2.8368`, `3019 -2.8368`, `198 -3.0243`
- request 1: `3019 -2.7174`, `1882 -2.8424`, `198 -3.0924`

This proves the active vLLM path is no longer silently falling back to baseline
attention. The values are close to the earlier HF FlexAttention reference probe,
but there is still no committed end-to-end numerical assertion against
`src/modeling_qwen3_batch_parscale.py`.

## Qwen3-0.6B Smoke Test

Command shape:

```bash
VLLM_NO_USAGE_STATS=1 XDG_CONFIG_HOME=/tmp/vllm-config .venv/bin/python - <<'PY'
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(model)
virtual_token_id = tok.encode(" reasoning", add_special_tokens=False)[-1]
prompts = [
    "Solve briefly: 2 + 2 = reasoning",
    "Solve briefly: 3 + 5 = reasoning",
]
params = [
    SamplingParams(
        temperature=0.0,
        max_tokens=2,
        extra_args={
            "cross_batch_attention": {
                "enabled": True,
                "group_id": "qwen3-smoke-2",
                "replica_id": i,
                "group_size": 2,
                "virtual_token_id": virtual_token_id,
                "virtual_window_size": 4,
            }
        },
    )
    for i in range(2)
]
llm = LLM(
    model=model,
    dtype="bfloat16",
    enforce_eager=True,
    max_model_len=256,
    max_num_seqs=2,
    gpu_memory_utilization=0.35,
    attention_config={"backend": "FLEX_ATTENTION"},
)
outs = llm.generate(prompts, params, use_tqdm=False)
for out in outs:
    print(out.request_id, repr(out.outputs[0].text), list(out.outputs[0].token_ids))
print("qwen3 cross-batch flex smoke ok")
PY
```

Result:

- Model loaded with `AttentionBackendEnum.FLEX_ATTENTION`.
- Generation completed with exit code `0`.
- Output:
  - request `0`: `' process\n'`, token ids `[1882, 198]`
  - request `1`: `' step by'`, token ids `[3019, 553]`
- The script printed `qwen3 cross-batch flex smoke ok`.
- vLLM logged `Engine core proc EngineCore died unexpectedly, shutting down
  client` during teardown after successful generation. A baseline Qwen3-0.6B
  FlexAttention run without cross-batch metadata also reproduces the same
  teardown log after successful generation, so this appears unrelated to the
  cross-batch implementation.

Earlier Qwen run without `VLLM_NO_USAGE_STATS=1` also generated successfully,
but a background usage-stat thread hit `PermissionError` writing
`/home/ubuntu/.config/vllm`. The rerun disabled usage stats and used a writable
config home.

## Current Confidence

This is a working prototype slice, not production-ready.

Approximate state: about 82% of the first milestone.

The current code proves:

- request metadata can enter through `SamplingParams.extra_args`
- V1 scheduler/worker/FlexAttention plumbing works for simple grouped requests
- FlexAttention mask semantics work for scalar and vectorized synthetic cases
- Cross-batch direct block-mask construction is a superset of slow-path block
  construction for synthetic grouped batches
- Cross-batch peer attention changes virtual-query attention outputs while
  leaving non-virtual query outputs identical in a dense reference test
- The Qwen3 batch-parscale absolute-position virtual causality rule is now
  reflected in the vLLM mask
- The active V1 `gpu_model_runner.py` path now forwards cross-batch metadata to
  FlexAttention
- A live Qwen3-0.6B baseline-vs-cross probe now shows material logit changes
  from peer virtual-token attention
- Group admission has a conservative KV-capacity preflight before moving a
  complete group to the scheduling front
- Cross-batch metadata is rejected for non-FlexAttention backends instead of
  silently being ignored
- Qwen3-0.6B can run grouped cross-batch FlexAttention smoke generation and a
  baseline-vs-cross logprob probe
- The trained non-dynamic checkpoint at
  `/home/ubuntu/parallel_reasoning/experiments/finetune/checkpoints/nb4_nr0.5_nv4_cbad0.1_2026-04-26-12-33-30/step_0001107_tokens_9M/adapter`
  loads in vLLM as a Qwen3-0.6B LoRA, including the `lm_head` LoRA path.
- vLLM now supports the trained HF position-scheduled virtual-token mode by
  setting `virtual_token_id=-1`; in that mode virtual positions are
  `virtual_window_size`, then every `virtual_window_size + 1` positions.
- A real-checkpoint probe with a 128-token prompt, `virtual_window_size=127`,
  four replicas, and first-token generation at the first virtual position now
  closely matches the HF reference after fixing the parent repo's custom Qwen3
  RoPE helper to read `config.rope_parameters["rope_theta"]`. The earlier HF
  reference used `config.rope_theta == 10000.0`, while stock Qwen3/vLLM use
  `rope_parameters["rope_theta"] == 1000000`.
  - HF top-1: token id `220` (`" "`), logprob `-0.039079`
  - vLLM top-1: token id `220` (`" "`), logprob `-0.039708`
  - Most overlapping top-token logprobs differ by about `0.00063`; the largest
    common top-token drift in the 20-token probe was about `0.06313`.
- The same trained adapter generated coherent arithmetic text in vLLM on
  `Problem: What is 7 + 15?\nSolution:`:
  `7 + 15 = 22\nSolution 2\n...`.
- Added an opt-in trained-adapter probe:
  `examples/offline_inference/cross_batch_trained_adapter_probe.py`.
  The HF mode should be run with the parent repo `.venv`; the vLLM mode should
  be run with the vLLM fork `.venv`; compare mode summarizes the JSON outputs.
- Follow-up isolation showed the original larger drift was in the parent repo's
  custom HF reference, not in vLLM. With cross-batch disabled, the custom
  `Qwen3BatchParScaleForCausalLM` path already diverged from stock
  `Qwen3ForCausalLM + PEFT`. The root cause was stale RoPE config handling in
  `src/shared.py`: it read `config.rope_theta` instead of Qwen3's active
  `config.rope_parameters["rope_theta"]`. After that fix, the trained adapter
  HF reference and vLLM first-virtual-position probe agree tightly.

The current code still does not prove:

- tight HF-reference numerical agreement across multi-token decode continuation
  for the trained adapter. The first virtual-position probe now matches closely,
  but continuation through later virtual positions still needs coverage.
- grouped scheduling is correct under every preemption, priority, async, and
  connector interaction

## Known Gaps

- A local A100-SXM4-80GB throughput sweep on 2026-04-26 shows the current
  prototype is correct but not yet performant at larger grouped batch sizes.
  Configuration: `Qwen/Qwen3-0.6B`, bf16, V1 FlexAttention,
  `enforce_eager=True`, prefix caching disabled, group size `4`,
  `virtual_token_id=-1`, `virtual_window_size=prompt_len-1`,
  `ignore_eos=True`, 3 measured repeats after warmup.

  Output-token throughput:

  | prompt | decode | groups | reqs | baseline tok/s | cross-batch tok/s |
  | --- | --- | --- | --- | ---: | ---: |
  | 128 | 64 | 1 | 4 | 278.86 | 263.36 |
  | 128 | 64 | 2 | 8 | 552.58 | 273.11 |
  | 128 | 64 | 4 | 16 | 1084.82 | 154.39 |
  | 128 | 256 | 1 | 4 | 275.23 | 243.59 |
  | 128 | 256 | 2 | 8 | 548.96 | 192.21 |
  | 128 | 256 | 4 | 16 | 994.36 | 104.00 |
  | 512 | 64 | 1 | 4 | 270.39 | 121.36 |
  | 512 | 64 | 2 | 8 | 490.98 | 78.04 |
  | 512 | 64 | 4 | 16 | 564.99 | 40.26 |
  | 512 | 256 | 1 | 4 | 279.66 | 124.73 |
  | 512 | 256 | 2 | 8 | 456.77 | 79.93 |
  | 512 | 256 | 4 | 16 | 515.04 | 41.16 |

  A short capacity probe at prompt `128`, decode `64`, and one measured repeat
  showed cross-batch runs at 5 groups / 20 requests (`150.33` output tok/s), but
  fails to compile at 6 groups / 24 requests and 8 groups / 32 requests with
  FlexAttention Triton resource errors:
  `Required: 197120 Hardware limit: 166912` at 6 groups and
  `Required: 221696 Hardware limit: 166912` at 8 groups. This is a backend
  kernel-resource limit, not KV capacity exhaustion.

- Scheduling now has a conservative KV-capacity preflight for grouped admission,
  but it is still not a true transactional allocation/rollback layer. Prefix
  cache, encoder, connector, and preemption interactions need broader tests.
- Running-group scheduling keeps live replicas together when they all fit, but
  more tests are needed for preemption, priority scheduling, async scheduling,
  and mixed grouped/ungrouped traffic.
- The request API is an `extra_args` research hook, not a polished public API.
- Only the V1 FlexAttention backend is implemented. Other attention backends
  reject cross-batch metadata.
- DBO/microbatching, DCP/DP, speculative decoding, CUDA graphs, prefix-cache
  sharing edge cases, sliding-window interactions, encoder-decoder models, and
  broad LoRA interaction are not validated.
- Eager mode is forced when cross-batch metadata is present. This avoids graph
  specialization problems and the first FlexAttention peer-topology shape
  problem, but leaves performance work unresolved.
- The HF-side `modeling_qwen3_batch_parscale.py` comparison is still ad hoc.
  It now covers the trained non-dynamic adapter at the first virtual position
  with close numerical agreement after the RoPE fix, but does not cover decode
  continuation across several virtual positions, padding, prefix cache, or more
  than one group.

## Next Steps

1. Broaden scheduler tests around priority scheduling, preemption,
   async scheduling, mixed grouped/ungrouped traffic, prefix-cache hits, and
   connector paths.
2. Extend the trained-adapter numerical probe beyond the first virtual position
   so it covers decode continuation, mixed replica prompts, padding, prefix
   cache settings, and multiple groups.
3. Convert the conservative KV-capacity preflight into a true transactional
   grouped allocation or rollback mechanism if broader scheduler tests expose a
   remaining split-group path.
4. Investigate the baseline FlexAttention Qwen3 engine-core teardown log if it
   matters for the surrounding benchmark harness.
