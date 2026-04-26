# vLLM Cross-Batch Attention Implementation State

Branch: `feature/cross-batch-flex-attention`

Implementation commit: `f32aa74b7` (`Add V1 cross-batch FlexAttention prototype`)

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
- `GPUModelRunner.prepare_inputs()` translates request-id keyed group metadata
  into worker batch-order tensors after the existing decode-first sort.
- `CommonAttentionMetadata` and `FlexAttentionMetadata` carry optional
  cross-batch tensors.
- FlexAttention now supports a cross-batch decoder mask:
  - same-request causal attention remains allowed
  - peer attention is allowed only for same-group virtual tokens
  - peer virtual attention is causal in virtual index
- FlexAttention direct block-mask construction includes peer KV blocks for
  enabled same-group peers, then relies on the mask for exact token filtering.
- Cross-batch batches force eager execution by overriding the batch descriptor to
  `CUDAGraphMode.NONE`.

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

Result: `7 passed`

```bash
python -m pytest tests/kernels/test_flex_attention.py -q \
  -k physical_to_logical_mapping_handles_reused_blocks
```

Result: `1 passed, 3 deselected`

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
  vllm/v1/worker/gpu/model_states/default.py \
  vllm/v1/attention/backend.py \
  vllm/v1/attention/backends/flex_attention.py \
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
  client` during teardown after successful generation. This should be
  investigated, but it did not prevent the smoke test from completing.

Earlier Qwen run without `VLLM_NO_USAGE_STATS=1` also generated successfully,
but a background usage-stat thread hit `PermissionError` writing
`/home/ubuntu/.config/vllm`. The rerun disabled usage stats and used a writable
config home.

## Current Confidence

This is a working prototype slice, not production-ready.

Approximate state: about 60% of the first milestone.

The current code proves:

- request metadata can enter through `SamplingParams.extra_args`
- V1 scheduler/worker/FlexAttention plumbing works for simple grouped requests
- FlexAttention mask semantics work for scalar and vectorized synthetic cases
- Qwen3-0.6B can run a grouped cross-batch FlexAttention smoke generation

The current code does not yet prove:

- logits are numerically correct against a reference implementation
- cross-batch peer attention changes exactly the intended virtual-token logits
- direct block-mask construction is equivalent to slow path for cross-batch
  cases
- group scheduling is fully transactional under KV allocation failure

## Known Gaps

- Scheduling is conservative but not fully atomic. If KV allocation fails after
  some replicas in a group are scheduled, the current implementation can still
  split the group. This needs transactional allocation or rollback.
- Running-group scheduling keeps live replicas together when they all fit, but
  more tests are needed for preemption, priority scheduling, async scheduling,
  and mixed grouped/ungrouped traffic.
- Virtual-position semantics currently use `logical_position %
  virtual_window_size`. This is only an approximation of the training-side
  semantics and may need a model-specific absolute virtual-position rule.
- The request API is an `extra_args` research hook, not a polished public API.
- Only the V1 FlexAttention backend is implemented. Other attention backends
  ignore the metadata.
- DBO/microbatching, DCP/DP, speculative decoding, CUDA graphs, prefix-cache
  sharing edge cases, sliding-window interactions, encoder-decoder models, and
  LoRA interaction are not validated.
- Eager mode is forced when cross-batch metadata is present. This avoids graph
  specialization problems but leaves performance work unresolved.
- No full end-to-end correctness test compares against the HF-side
  `modeling_qwen3_batch_parscale.py` behavior.

## Next Steps

1. Add a synthetic FlexAttention direct-vs-slow block-mask test for cross-batch
   metadata that avoids gated Hugging Face models.
2. Add a tiny-model logits test that demonstrates disabled metadata matches
   baseline and enabled metadata changes only virtual-token grouped paths.
3. Make scheduler group admission transactional around KV allocation.
4. Replace modulo virtual indexing with the exact absolute virtual-position rule
   required by the Qwen3 batch-parscale model.
5. Add a Qwen3-specific integration test once the exact virtual-token convention
   and expected logits/reference behavior are defined.

