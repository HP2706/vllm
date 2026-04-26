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

Output-token throughput, original peer-loop prototype:

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

Output-token throughput after replacing the request-by-request peer matrix with
group-scoped physical-block lookup:

| prompt | decode | groups | reqs | crossbatch physical tok/s | flex tok/s | flashattn tok/s | cross/flex | cross/flash |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 64 | 1 | 4 | 255.82 | 280.22 | 420.33 | 91.29% | 60.86% |
| 128 | 64 | 2 | 8 | 498.16 | 543.52 | 839.05 | 91.65% | 59.37% |
| 128 | 64 | 4 | 16 | 746.54 | 1082.76 | 1643.38 | 68.95% | 45.43% |
| 128 | 256 | 1 | 4 | 262.17 | 276.11 | 428.74 | 94.95% | 61.15% |
| 128 | 256 | 2 | 8 | 445.37 | 553.12 | 851.05 | 80.52% | 52.33% |
| 128 | 256 | 4 | 16 | 532.34 | 998.79 | 1690.48 | 53.30% | 31.49% |
| 512 | 64 | 1 | 4 | 182.10 | 272.04 | 435.24 | 66.94% | 41.84% |
| 512 | 64 | 2 | 8 | 222.08 | 494.18 | 838.20 | 44.94% | 26.50% |
| 512 | 64 | 4 | 16 | 233.87 | 566.20 | 1589.73 | 41.31% | 14.71% |
| 512 | 256 | 1 | 4 | 184.12 | 278.89 | 418.27 | 66.02% | 44.02% |
| 512 | 256 | 2 | 8 | 219.24 | 457.70 | 832.12 | 47.90% | 26.35% |
| 512 | 256 | 4 | 16 | 229.61 | 515.45 | 1631.70 | 44.54% | 14.07% |

Capacity notes:

- Crossbatch ran at 5 groups / 20 requests for prompt 128, decode 64:
  `150.33` output tok/s.
- Crossbatch failed to compile at 6 groups / 24 requests on A100:
  `Required: 197120 Hardware limit: 166912`.
- Crossbatch also failed at 8 groups / 32 requests:
  `Required: 221696 Hardware limit: 166912`.
- These failures are FlexAttention Triton kernel-resource limits, not KV-cache
  capacity exhaustion.
- After the physical-block lookup change, crossbatch ran at 6 groups / 24
  requests for prompt 128, decode 64: `1103.27` output tok/s.

Raw result files:

- `qwen3_0_6b_a100_flex_crossbatch_2026-04-26.json`
- `qwen3_0_6b_a100_flex_crossbatch_physical_lookup_2026-04-26.json`
- `qwen3_0_6b_a100_flashattn_baseline_2026-04-26.json`
- `qwen3_0_6b_a100_crossbatch_capacity_g5_2026-04-26.json`
- `qwen3_0_6b_a100_crossbatch_capacity_g6_failed_2026-04-26.json`
- `qwen3_0_6b_a100_crossbatch_capacity_g7_failed_2026-04-26.json`
