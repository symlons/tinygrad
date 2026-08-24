# A100 BF16 GEMM sweep — 2026-08-21

Fresh, contemporaneous comparison on an NVIDIA A100-SXM4-40GB. Shapes are
written unambiguously as `MxK @ KxN`. All paths use BF16 inputs, FP32
accumulation, and BF16 output. cuDNN is 9.20.0 through
`nvidia-cudnn-frontend` 1.27.0.

The runner uses 5000 warmups/300 iterations below 20 GFLOP, 3000/200 below
300 GFLOP, and 500/100 otherwise. This keeps the A100 at boost for short
kernels. Results are raw GPU execution time, excluding graph construction.

## Historical 13-shape sweep

| Name | GEMM | custom | cuBLAS | cuDNN | custom/cuBLAS |
|---|---|---:|---:|---:|---:|
| square-1024 | `1024x1024 @ 1024x1024` | 65.16 | 85.83 | 75.36 | 75.9% |
| square-2048 | `2048x2048 @ 2048x2048` | 152.28 | 159.57 | 159.49 | 95.4% |
| square-4096 | `4096x4096 @ 4096x4096` | 252.35 | 251.62 | 251.59 | **100.3%** |
| square-8192 | `8192x8192 @ 8192x8192` | 267.68 | 268.46 | 266.84 | 99.7% |
| wide-14336 | `4096x4096 @ 4096x14336` | 252.47 | 254.56 | 252.71 | 99.2% |
| deep-14336 | `4096x14336 @ 14336x4096` | 251.45 | 260.08 | 257.17 | 96.7% |
| wide-28672 | `8192x8192 @ 8192x28672` | 259.45 | 264.92 | 264.61 | 97.9% |
| deep-28672 | `8192x28672 @ 28672x8192` | 263.79 | 260.03 | 260.74 | **101.4%** |
| short-m-2048 | `2048x8192 @ 8192x8192` | 249.08 | 257.96 | 256.83 | 96.6% |
| short-n-2048 | `8192x8192 @ 8192x2048` | 252.11 | 257.83 | 258.65 | 97.8% |
| thin-k-128 | `8192x128 @ 128x8192` | 89.83 | 98.21 | 98.46 | 91.5% |
| thin-n-128 | `8192x8192 @ 8192x128` | 116.09 | 124.57 | 124.50 | 93.2% |
| thin-m-256 | `256x8192 @ 8192x8192` | 169.13 | 204.20 | 203.64 | 82.8% |

The geometric-mean custom/cuBLAS ratio is **94.2%**. The custom kernel wins
two shapes, is at least 95% on eight, and has three material gaps: 1024³,
thin K, and thin M. The new thin-N kernel is now at 93.2%.

## Added diagnostic and production shapes

These 13 shapes are now part of `ADDED` in `sweep_bf16_gemm.py`.

| Name | GEMM | custom | cuBLAS | cuDNN | custom/cuBLAS |
|---|---|---:|---:|---:|---:|
| square-512 | `512x512 @ 512x512` | 10.12 | 18.92 | 8.98 | 53.5% |
| k-transition-256 | `8192x256 @ 256x8192` | 138.74 | 178.11 | 177.95 | 77.9% |
| k-transition-512 | `8192x512 @ 512x8192` | 182.98 | 215.44 | 211.75 | 84.9% |
| k-transition-768 | `8192x768 @ 768x8192` | 206.49 | 228.08 | 226.70 | 90.5% |
| k-transition-1024 | `8192x1024 @ 1024x8192` | 221.82 | 236.32 | 233.70 | 93.9% |
| thin-n-256 | `8192x8192 @ 8192x256` | 169.92 | 205.90 | 205.63 | 82.5% |
| thin-n-512 | `8192x8192 @ 8192x512` | 171.51 | 208.73 | 206.02 | 82.2% |
| thin-m-512 | `512x8192 @ 8192x8192` | 171.91 | 210.27 | 207.98 | 81.8% |
| thin-m-1024 | `1024x8192 @ 8192x8192` | 221.78 | 245.66 | 243.79 | 90.3% |
| cta-boundary-104 | `3328x8192 @ 8192x1024` | 261.29 | 259.75 | 256.15 | **100.6%** |
| cta-boundary-112 | `3584x8192 @ 8192x1024` | 150.70 | 191.33 | 190.88 | 78.8% |
| llm-wide-11008 | `4096x4096 @ 4096x11008` | 255.01 | 258.55 | 258.44 | 98.6% |
| llm-deep-11008 | `4096x11008 @ 11008x4096` | 258.86 | 260.20 | 260.17 | 99.5% |

The 11008 projection shapes should remain permanently: both are realistic
and both sit within 1.5% of cuBLAS. The K-transition points should remain to
guard persistent dispatch. The thin-M/N points and CTA boundary pair should
remain as optimization diagnostics rather than headline geomean inputs.

## Profiling of the added shapes

Nsight Compute hardware counters are disabled by the host
(`ERR_NVGPUCTRPERM`), so bank-conflict, cache-hit, and SM-throughput counters
could not be collected. Nsight Systems CUDA tracing was available and
reported exact launch/resource data. Durations below are traced CUDA-runtime
launches after one warmup; use the stable sweep above for performance ratios.

The four historical outliers have distinct launch profiles:

| Name | dispatch | grid CTAs | threads | regs/thread | dynamic shared | local/thread | traced duration |
|---|---|---:|---:|---:|---:|---:|---:|
| square-1024 | flat | 32 | 256 | 236 | 96 KiB | 0 | 42 us |
| thin-k-128 | persistent | 108 | 256 | 254 | 96 KiB | 0 | 247 us |
| thin-n-128 | flat, 128x128 ILP | 64 | 128 | 254 | 96 KiB | 0 | 189 us |
| thin-m-256 | flat | 64 | 256 | 236 | 96 KiB | 0 | 262 us |

Thus 1024³ is a 32-CTA fixed-cost/underfill problem; thin K already fills all
108 SMs through persistence but does too little work per CTA; thin N and thin
M both launch only 64 CTAs. The thin-N specialization improves per-CTA ILP
with four warps, but cannot create more than 64 output tiles.

| Name | dispatch | grid CTAs | waves/108 SMs | threads | regs/thread | dynamic shared | local/thread | traced duration |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| square-512 | persistent | 108 | 1.00 | 256 | 248 | 96 KiB | 0 | 27 us |
| k-transition-256 | persistent | 108 | 1.00 | 256 | 248 | 96 KiB | 0 | 317 us |
| k-transition-512 | persistent | 108 | 1.00 | 256 | 248 | 96 KiB | 0 | 487 us |
| k-transition-768 | flat | 2048 | 18.96 | 256 | 236 | 96 KiB | 0 | 636 us |
| k-transition-1024 | flat | 2048 | 18.96 | 256 | 236 | 96 KiB | 0 | 785 us |
| thin-n-256 | flat | 64 | 0.59 | 256 | 236 | 96 KiB | 0 | 262 us |
| thin-n-512 | flat | 128 | 1.19 | 256 | 236 | 96 KiB | 0 | 517 us |
| thin-m-512 | flat | 128 | 1.19 | 256 | 236 | 96 KiB | 0 | 517 us |
| thin-m-1024 | flat | 256 | 2.37 | 256 | 236 | 96 KiB | 0 | 778 us |
| cta-boundary-104 | flat | 104 | 0.96 | 256 | 236 | 96 KiB | 0 | 263 us |
| cta-boundary-112 | flat | 112 | 1.04 | 256 | 236 | 96 KiB | 0 | 515 us |
| llm-wide-11008 | flat | 1376 | 12.74 | 256 | 236 | 96 KiB | 0 | 1742 us |
| llm-deep-11008 | flat | 512 | 4.74 | 256 | 236 | 96 KiB | 0 | 1723 us |

The dominant new profile finding is the **wave quantization cliff**. At 104
CTAs, all work completes in one wave and the custom kernel slightly beats
cuBLAS. At 112 CTAs, only four CTAs spill into a second wave, but traced
duration nearly doubles from 263 to 515 us. cuBLAS and cuDNN also slow down,
but the custom kernel's cliff is larger. Thin-N=512 and thin-M=512 have the
same 128-CTA grid, almost identical traced duration, and almost identical
throughput; this identifies a symmetric output-tile occupancy problem.

The persistent transition is also visible: K<=512 launches 108 CTAs at 248
registers/thread; K>=768 launches 2048 flat CTAs at 236 registers/thread.
Performance rises smoothly across the threshold, so the current 512 cutoff
is directionally correct, though K=544 and K=640 are worthwhile fine-grained
follow-ups if the dispatch heuristic is tuned further.

## Reproduction

```bash
uv run --with nvidia-cudnn-frontend==1.27.0 \
  python -m extra.gemm.sweep_bf16_gemm --set all --backend all --json /tmp/bf16-sweep.json
```

The raw Nsight Systems reports for this run were written under
`/tmp/bf16_profiles/`; they are intentionally not committed because they are
large binary, machine-specific artifacts.
