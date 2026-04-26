# Qwen3-0.6B Attention Backend Throughput, A100

Date: 2026-04-26

Hardware and software:

- GPU: NVIDIA A100-SXM4-80GB
- Model: `Qwen/Qwen3-0.6B`
- dtype: bf16
- vLLM: V1 engine, local `feature/cross-batch-flex-attention`
- `enforce_eager=True`
- `enable_prefix_caching=False`
- `ignore_eos=True`
- group size: 4
- repeats: 3 measured repeats after one warmup per case

Benchmarks:

- `crossbatch`: FlexAttention plus custom cross-batch mask, with
  `virtual_token_id=-1` and `virtual_window_size=prompt_len-1`.
- `flex`: normal non-cross-batch FlexAttention baseline.
- `flashattn`: normal non-cross-batch FlashAttention v2 baseline.

Output-token throughput:

| prompt | decode | groups | reqs | crossbatch tok/s | flex tok/s | flashattn tok/s | flash/flex | cross/flash |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 64 | 1 | 4 | 263.36 | 278.86 | 420.33 | 1.51x | 62.66% |
| 128 | 64 | 2 | 8 | 273.11 | 552.58 | 839.05 | 1.52x | 32.55% |
| 128 | 64 | 4 | 16 | 154.39 | 1084.82 | 1643.38 | 1.51x | 9.39% |
| 128 | 256 | 1 | 4 | 243.59 | 275.23 | 428.74 | 1.56x | 56.81% |
| 128 | 256 | 2 | 8 | 192.21 | 548.96 | 851.05 | 1.55x | 22.58% |
| 128 | 256 | 4 | 16 | 104.00 | 994.36 | 1690.48 | 1.70x | 6.15% |
| 512 | 64 | 1 | 4 | 121.36 | 270.39 | 435.24 | 1.61x | 27.88% |
| 512 | 64 | 2 | 8 | 78.04 | 490.98 | 838.20 | 1.71x | 9.31% |
| 512 | 64 | 4 | 16 | 40.26 | 564.99 | 1589.73 | 2.81x | 2.53% |
| 512 | 256 | 1 | 4 | 124.73 | 279.66 | 418.27 | 1.50x | 29.82% |
| 512 | 256 | 2 | 8 | 79.93 | 456.77 | 832.12 | 1.82x | 9.61% |
| 512 | 256 | 4 | 16 | 41.16 | 515.04 | 1631.70 | 3.17x | 2.52% |

Capacity notes:

- Crossbatch ran at 5 groups / 20 requests for prompt 128, decode 64:
  `150.33` output tok/s.
- Crossbatch failed to compile at 6 groups / 24 requests on A100:
  `Required: 197120 Hardware limit: 166912`.
- Crossbatch also failed at 8 groups / 32 requests:
  `Required: 221696 Hardware limit: 166912`.
- These failures are FlexAttention Triton kernel-resource limits, not KV-cache
  capacity exhaustion.

Raw result files:

- `qwen3_0_6b_a100_flex_crossbatch_2026-04-26.json`
- `qwen3_0_6b_a100_flashattn_baseline_2026-04-26.json`
- `qwen3_0_6b_a100_crossbatch_capacity_g5_2026-04-26.json`
- `qwen3_0_6b_a100_crossbatch_capacity_g6_failed_2026-04-26.json`
- `qwen3_0_6b_a100_crossbatch_capacity_g7_failed_2026-04-26.json`
