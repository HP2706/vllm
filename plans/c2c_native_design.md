# RFC: Native Cross-Model KV Transfer & Fusion in vLLM

Status: draft (design for upstream discussion)
Branch: `c2c-kv-transfer`
Prior art: prototype `C2CConnector` (this branch), measured on OpenBookQA-500:
receiver-alone 39.8% → in-engine C2C fusion 49.8% (offline reference 52.6%),
model-switch cost ~81 ms vs ~990 ms text handoff.

## Problem

vLLM's KV transfer subsystem (disaggregated prefill) assumes source and
destination are **the same model**: the NIXL connector's compatibility hash
hard-fails on `num_kv_heads`, `head_size`, `num_hidden_layers`
(`nixl_connector.py:219-229`). There is no way to move KV between *different*
models, and no way to apply a learned transformation (projection, fusion,
quantization) in flight. Research like Cache-to-Cache (arXiv:2510.03215) shows
cross-model KV fusion beats text handoff by double-digit accuracy at ~12x lower
switch latency, but today it only exists as offline HF wrappers.

## Key enabling fact (from code research)

In v1, FlashAttention-class backends write the current chunk's K/V into the
paged cache **before** the attention kernel reads it back:

- `attention.py:438-468`: `unified_kv_cache_update(...)` (writes via
  `reshape_and_cache_flash`) is a separate custom op from
  `unified_attention_with_output(...)` (reads `key_cache`/`value_cache` —
  `flash_attn.py:626-760` passes the paged tensors to
  `flash_attn_varlen_func`).

So there is a well-defined per-layer window where externally-derived KV can be
fused into the cache for the *current* tokens with **exact** semantics: the
same layer's attention, and every subsequent token, sees fused KV. (The
offline C2C implementation needs a second prefill pass to achieve this; vLLM
gets it in one.) FlexAttention computes attention from fresh tensors
(`forward_includes_kv_cache_update = False`), so the feature must be gated
per-backend via a capability flag.

## Proposed architecture: three composable primitives

### P1. Per-layer KV post-write hook ("KVMutationPass")

A new optional hook on the worker connector API, called between cache write
and kernel read for each attention layer:

```python
class KVConnectorBase_V1:
    # NEW. Called after reshape_and_cache for `layer_name`, before the
    # attention kernel reads the paged cache. May mutate kv_cache slots
    # of the current step's tokens (slot_mapping in attn_metadata).
    def mutate_kv_post_write(
        self, layer_name: str, kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> None: ...
```

Wiring: extend the `@maybe_transfer_kv_layer` mechanism
(`kv_transfer_utils.py:14-61`) so the unified-attention path becomes
`update → mutate_kv_post_write → attention`. Costs nothing when no connector
implements it (same guard pattern as today: `has_connector_metadata()`).

Compatibility:
- CUDA graphs: layers with an active mutation pass force
  `CUDAGraphMode.NONE` for that batch, exactly like this fork's cross-batch
  attention already does (`gpu_model_runner.py:3738`,
  `force_eager=...`). Piecewise graphs can be supported later because the
  hook sits at an existing custom-op boundary.
- Chunked prefill: the hook fires per step with that step's
  `slot_mapping`; fusion is pointwise per position, so chunked prompts work
  with zero extra bookkeeping (each chunk fuses its own slots).
- Backends: gated by `AttentionBackend.reads_current_tokens_from_cache`
  capability; FlashAttention/Triton yes, FlexAttention no (falls back to
  post-forward fusion with documented approximate semantics).

This primitive is independently useful upstream: KV quantization-on-write,
cache watermarking/debugging, eviction research, and this fork's cross-batch
virtual-token work could all use it.

### P2. KVTransform: heterogeneous transfer as a declared capability

Replace "same model or refuse" with a structured handshake. The NIXL
compatibility hash splits into:

- transport compatibility (block layout, dtype, NIXL version) — still a hash;
- **model signature** (layer count, kv heads, head dim, RoPE params) —
  exchanged as data in `NixlAgentMetadata`, not hashed.

A `KVTransform` plugin declares what signature pair it bridges:

```python
class KVTransform(ABC):
    source_signature: ModelKVSignature   # e.g. qwen2.5-0.5b: 24L, 2H, 64d
    target_signature: ModelKVSignature   # e.g. qwen3-0.6b: 28L, 8H, 128d
    layer_map: dict[int, int]            # target layer -> source layer

    @abstractmethod
    def transform(
        self, tgt_layer: int,
        source_kv: torch.Tensor,   # [2, N, H_s, D_s] post-RoPE cache space
        target_kv: torch.Tensor,   # [2, N, H_t, D_t] receiver's own KV
    ) -> torch.Tensor: ...
```

Identity transform == today's disagg prefill. `C2CFuserTransform` wraps the
trained projectors (pointwise residual fusion in post-RoPE cache space, which
is exactly what the paged cache stores — verified against thu-nics/C2C's
injection point). Registration mirrors the connector factory
(`kv_transform_module_path` for out-of-tree, named registry in-tree).

Transfer flow (receiver side), reusing the **existing async machinery**:

1. `get_num_new_matched_tokens` returns `(0, load_kv_async=True)` variant:
   request enters `WAITING_FOR_REMOTE_KVS` (`scheduler.py:792-811`) while the
   sharer KV is pulled via NIXL one-sided reads into a registered staging
   ring buffer (NOT into the paged cache — the receiver still computes its
   own prefill; C2C is residual fusion, not cache substitution).
2. When `get_finished` reports the staging landed, the request is promoted
   (`scheduler.py:2258-2273`) and scheduled for normal prefill.
3. During prefill, P1's `mutate_kv_post_write` runs the transform per layer:
   read staged source KV for `layer_map[l]`, fuse with the just-written
   target KV at this step's slots, write back.
4. Failures reuse `kv_load_failure_policy`: a failed staging load simply
   degrades to a receiver-only prefill (fusion skipped), which is strictly
   the no-C2C baseline — a much friendlier failure mode than disagg's.

Request pairing is explicit via `kv_transfer_params` (the NIXL pattern,
already plumbed API → `Request.extra_args` → response,
`request.py:109-112`, `scheduler.py:1599-1648`): the sharer's response
returns `{engine_id, request_id, model_signature}`; the receiver's request
carries it back. No content-hash rendezvous.

Two-GPU / multi-node story: inherited entirely from NIXL. Sharer engine on
GPU0, receiver engine on GPU1 of the same node → UCX picks CUDA IPC / NVLink;
different nodes → GPUDirect RDMA. Heterogeneous TP between the two engines
already works in NIXL (head-split rank mapping, `utils.py:321-485`); the
transform runs after a per-rank gather when the source TP shard doesn't cover
the heads the projector needs (for small sharers, TP=1 source avoids this
entirely).

### P3. Transform weights as managed adapters

The C2C fuser (~1 GB of per-layer MLPs per model pair) loads like a LoRA
adapter, not like a baked-in model component:

- `--kv-transforms qwen3-0.6b+qwen2.5-0.5b=nics-efc/C2C_Fuser/...` CLI,
  hot load/unload, LRU-cached on the worker (pattern:
  `lora/model_manager.py:62-150`, minus Punica — transforms run as plain
  batched modules, one compiled `nn.Module` over all layers).
- Selected per request via `kv_transfer_params["transform"]`; signature pair
  verified against the handshake at request time, not engine start.

## Alternative considered: in-engine sharer (spec-decode pattern)

Host the sharer inside the receiver's engine the way speculative decoding
hosts a draft model (`DraftModelProposer`, `gpu_model_runner.py:502-566`):
sharer forward runs on the same scheduled prefill tokens, its KV staged
transiently, fusion via P1. No second server, no transport, lowest latency.

Rejected as the *first* upstream target because the draft-model path carries
restrictive constraints we'd inherit and have to relax one by one: same TP as
target enforced (`draft_model.py:36-51`, torch.compile cache collisions),
last-PP-rank same-device placement only, vocab-size assertions, and no
per-request enable/disable. It's the right *second* step — P1 and P3 are
shared infrastructure, and a `SharerRunner` can be added behind
`SpeculativeConfig`-style config later for the single-node agent ping-pong
use case.

## Cross-tokenizer C2C design notes

The clean v1 target is same-tokenizer C2C because KV cache fusion is not
tokenizer-invariant. A token-level KL or token-level cache transfer is only
well-defined when the two models agree on positions and vocabulary. With
different tokenizers, both models still define distributions and internal
states over the same underlying object: text/UTF-8 byte strings. The alignment
problem therefore moves from "compare token IDs" to "align cache positions by
text spans."

The statistically clean view is sequence-level: sample a text continuation
from one model, score the same text under both models using each model's own
tokenizer, and estimate KL over strings. This is valid for evaluation, but it
does not give the dense per-position signal needed for useful KV-cache fusion.
For C2C, the practical common substrate should be byte spans or synchronized
text chunks.

### Prefill-only cross-tokenizer transfer

The lowest-risk cross-tokenizer extension is prompt-prefill transfer:

1. Run the sharer and receiver over the same raw prompt text, each with its
   own tokenizer.
2. Track the byte span for every token:
   `receiver token i -> [a_i, b_i)`, `sharer token j -> [c_j, d_j)`.
3. For each receiver position, collect all sharer positions whose byte spans
   overlap or exactly cover the receiver span.
4. Pool the corresponding sharer K/V vectors into one aligned source vector
   for the receiver position. Candidate pooling strategies: mean pooling,
   last-token pooling, span-length-weighted pooling, or a small learned
   attention/span encoder.
5. Feed the pooled source K/V plus the receiver's just-written K/V into the
   `KVTransform`, then write the fused result back through P1's
   `mutate_kv_post_write` hook.

This preserves causality as long as the aligned sharer span never includes
bytes beyond the receiver prefix. Exact-cover chunks are preferable to raw
overlap for correctness: form minimal synchronized chunks where both tokenizers
land on the same byte boundary, pool within each chunk, then distribute the
chunk representation to the receiver positions inside that chunk.

### Generation-time transfer is harder

Ongoing cache exchange during decoding is much less direct. If the sharer and
receiver are both generating, their token boundaries and sometimes their text
continuations will diverge. A safe design needs explicit synchronization
points over emitted bytes/chunks, not token positions. Between synchronization
points, the receiver should either run alone or consume only source states that
are known to correspond to an already-materialized shared text prefix.

That makes "prefill-only C2C" the right first milestone for mismatched
tokenizers. A later generation-time version should probably use chunk-level
handoff:

- accumulate generated bytes until both tokenizers reach a shared boundary;
- align the completed chunk;
- transfer/fuse only the source K/V states that causally correspond to that
  chunk;
- resume receiver decoding from the fused chunk boundary.

This is slower and less elegant than same-tokenizer C2C, but it is still much
less lossy than text-to-text handoff if the source cache carries useful latent
state.

### RoPE and position handling

RoPE makes naive cross-tokenizer cache transfer especially fragile. Stored keys
are position-rotated in the source model's token coordinate system. If a text
span is source position 7 but receiver position 5, the key is not directly in
the receiver's positional frame. The simple learned-projector approach may
absorb some of this, but a principled implementation should expose position
metadata to the transform:

- source token positions and receiver token positions;
- byte-span offsets;
- RoPE configuration for both models.

For RoPE models, the transform can either learn the correction or explicitly
de-rotate source keys from source positions and re-rotate them into receiver
positions before fusion. Values do not carry RoPE rotation, but they are still
layer/model-specific and need projection.

### Possible statistical interfaces

Relevant approaches from the cross-tokenizer literature map naturally onto
different C2C designs:

- **String-level KL / Monte Carlo KL:** correct for evaluation and model
  comparison, but too sparse for cache fusion.
- **Byte-level interface:** convert tokenized model behavior to byte-level
  probabilities/states, then fuse over a shared byte stream. This is the most
  principled long-term route, especially for arbitrary tokenizers.
- **Chunk alignment:** align minimal text chunks shared by both tokenizers and
  pool K/V inside each chunk. This is probably the most practical vLLM path.
- **Optimal transport over vocabularies/logits:** useful for distillation
  losses, but less obviously correct for KV caches because cache states are
  position- and layer-dependent, not just output distributions.
- **Cross-tokenizer likelihood scoring:** attractive for training objectives,
  because it recovers likelihood ratios across tokenizers and can supervise a
  cache fuser without pretending token IDs match.

### Training objective for mismatched tokenizers

Do not try to train cross-tokenizer C2C by directly imposing token-level KL
between the models. The receiver's output space is its own vocabulary. Better
objectives:

- receiver next-token cross-entropy on the target text, conditioned on fused
  cache;
- sequence-level teacher likelihood or reward on the receiver's generated
  text;
- downstream task loss, as in the current C2C evaluation setup;
- optional auxiliary alignment loss between pooled source spans and receiver
  hidden/cache states.

The base models should remain frozen at first. Train only the span aligner,
pooler, layer mapping/gates, and K/V projectors. This keeps the experiment
close to C2C's adapter-like deployment model and makes failure degrade cleanly
to receiver-only inference.

### vLLM implications

Supporting mismatched tokenizers would require extending the P2 handshake:

- include tokenizer identity and normalization settings in `ModelKVSignature`;
- expose per-token byte spans from each engine's tokenizer path;
- transfer source K/V plus source position/span metadata into the staging ring;
- let `KVTransform` consume an alignment map, not just same-length source and
  target K/V tensors.

The transform signature would need to evolve from positionwise same-length
fusion to span-aware fusion:

```python
class SpanAlignedKVTransform(KVTransform):
    def transform(
        self,
        tgt_layer: int,
        source_kv: torch.Tensor,        # [2, N_src, H_s, D_s]
        target_kv: torch.Tensor,        # [2, N_tgt, H_t, D_t]
        alignment: TokenSpanAlignment,  # maps target positions/chunks to source spans
        source_positions: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> torch.Tensor: ...
```

This should remain a later milestone. Same-tokenizer C2C validates the core
vLLM hook, staging, adapter loading, and performance story. Cross-tokenizer C2C
then becomes an additional transform capability rather than a prerequisite for
the native design.

## What stays out of scope

- Cross-tokenizer alignment (C2C's TokenAligner): v1 requires
  shared-vocab pairs (Qwen↔Qwen etc.); the handshake rejects mismatched
  vocabs loudly.
- Prefix-cache-aware fusion dedup across requests: correct already
  (fusion is pointwise + causal), but cache-hit prefixes fused under a
  *different* pairing key should invalidate; v1 punts by documenting that
  fused blocks are keyed like normal blocks.
- Encoder-decoder, MLA, hybrid-attention models.

## PR sequencing (each lands independently)

1. **PR-1 (small, no behavior change):** `mutate_kv_post_write` hook +
   backend capability flag + eager-mode forcing when active. Tests: a toy
   connector that zeroes a slot and asserts the same layer's attention saw it.
2. **PR-2:** `KVTransform` interface + registry + identity transform; NIXL
   handshake split (transport hash vs model signature data). Tests: identity
   transform reproduces existing disagg-prefill numerics bit-exact.
3. **PR-3:** staging-ring path for "transfer without cache substitution" +
   `WAITING_FOR_REMOTE_KVS` integration + failure-degrades-to-baseline.
4. **PR-4:** adapter-style transform weight management (+ CLI), C2C fuser as
   the reference non-identity transform, with an accuracy regression test
   against the offline implementation on a 50-question OpenBookQA slice.
5. **Later:** in-engine `SharerRunner` (spec-decode pattern) reusing P1/P3.

## Performance notes (from the prototype)

- Transfer is never the bottleneck: 0.01–0.12 ms (up to 870 GB/s same-GPU
  IPC; NIXL adds ~µs-scale posting overhead).
- Fusion currently costs 38–60 ms because 28 projectors run sequentially in
  eager mode; one batched/compiled module should bring it to ~5 ms, making
  total switch overhead ≈ sharer prefill time.
- Staging memory for the sharer KV of an 8k-token prompt is ~100 MB (bf16,
  0.5B sharer) — bounded by the ring buffer, accounted against
  `gpu_memory_utilization` at profiling time (unlike the prototype's
  unaccounted dict).
