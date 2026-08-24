# A100 BF16 GEMM sweep — 2026-08-24

Fresh full-set rerun (`--set all --backend all`, all 26 shapes from
`sweep_bf16_gemm.py`) on the same A100-SXM4-40GB, three days after the
2026-08-21 sweep, to check for regressions and measure the day's kernel
change. No kernel code had changed yet when this sweep was collected (see
"1024³ fix" section below for the one change made afterward).

## Infra fix that made this sweep practical: compile cache was on network storage

The sweep initially stalled for minutes at a time on `D (disk sleep)`
(`folio_wait_bit_common`), with 0% GPU utilization — not a compute or GPU
hang. Root cause: tinygrad's kernel compile cache
(`~/.cache/tinygrad/cache.db`, a SQLite DB, gated by `CCACHE`, default on)
lives under `$XDG_CACHE_HOME`, which on this cluster resolves to
`/cluster/home/...` — a **CephFS network filesystem** (`mount` confirms
`type ceph`). SQLite writes over a network filesystem are slow (locking/fsync
round trips), and every new (M, N, K, config) combination in the sweep
compiles a genuinely new kernel, so this was hit on nearly every shape.

**Fix**: point `XDG_CACHE_HOME` at local disk for the duration of GPU work —
`/tmp` and `/` are both local `ext4` on this node (`/raid` also local,
mount-confirmed), unlike `/cluster/home`. With
`XDG_CACHE_HOME=/tmp/<user>-tinygrad-cache`, the same sweep that stalled for
8+ minutes on shape 3/26 completed all 26 shapes in about 30 minutes total.
This is a per-session environment setting, not a code change — worth setting
before any GPU benchmarking work on this cluster. See
[[slurm_gpu_access]]-adjacent note in memory.

## Full comparison vs. 2026-08-21 baseline

`custom` and `cuBLAS`/`cuDNN` in TFLOPS; `Δcustom` and `ΔcuBLAS` are today's
value minus 2026-08-21's, as a percentage of the old value.

| Name | custom (old→new) | Δcustom | cuBLAS (old→new) | ΔcuBLAS | custom/cuBLAS |
|---|---|---:|---|---:|---:|
| square-1024 | 65.16→65.49 | +0.5% | 85.83→86.47 | +0.7% | 75.7% |
| square-2048 | 152.28→151.90 | -0.2% | 159.57→159.22 | -0.2% | 95.4% |
| square-4096 | 252.35→250.18 | -0.9% | 251.62→246.77 | -1.9% | **101.4%** |
| square-8192 | 267.68→259.99 | -2.9% | 268.46→262.66 | -2.2% | 99.0% |
| wide-14336 | 252.47→247.65 | -1.9% | 254.56→253.88 | -0.3% | 97.5% |
| deep-14336 | 251.45→250.95 | -0.2% | 260.08→257.35 | -1.0% | 97.5% |
| wide-28672 | 259.45→253.74 | -2.2% | 264.92→259.72 | -2.0% | 97.7% |
| deep-28672 | 263.79→258.68 | -1.9% | 260.03→257.10 | -1.1% | **100.6%** |
| short-m-2048 | 249.08→246.31 | -1.1% | 257.96→250.46 | -2.9% | 98.3% |
| short-n-2048 | 252.11→247.10 | -2.0% | 257.83→250.54 | -2.8% | 98.6% |
| thin-k-128 | 89.83→89.60 | -0.3% | 98.21→97.53 | -0.7% | 91.9% |
| thin-n-128 | 116.09→116.36 | +0.2% | 124.57→125.68 | +0.9% | 92.6% |
| thin-m-256 | 169.13→167.13 | -1.2% | 204.20→200.66 | -1.7% | 83.3% |
| square-512 | 10.12→10.18 | +0.6% | 18.92→19.08 | +0.8% | 53.4% |
| k-transition-256 | 138.74→139.00 | +0.2% | 178.11→175.50 | -1.5% | 79.2% |
| k-transition-512 | 182.98→179.88 | -1.7% | 215.44→210.46 | -2.3% | 85.5% |
| k-transition-768 | 206.49→202.64 | -1.9% | 228.08→222.18 | -2.6% | 91.2% |
| k-transition-1024 | 221.82→213.99 | -3.5% | 236.32→231.89 | -1.9% | 92.3% |
| thin-n-256 | 169.92→167.78 | -1.3% | 205.90→202.30 | -1.7% | 82.9% |
| thin-n-512 | 171.51→169.60 | -1.1% | 208.73→206.72 | -1.0% | 82.0% |
| thin-m-512 | 171.91→171.17 | -0.4% | 210.27→201.25 | -4.3% | 85.1% |
| thin-m-1024 | 221.78→216.90 | -2.2% | 245.66→240.18 | -2.2% | 90.3% |
| cta-boundary-104 | 261.29→257.72 | -1.4% | 259.75→254.76 | -1.9% | **101.2%** |
| cta-boundary-112 | 150.70→150.58 | -0.1% | 191.33→189.19 | -1.1% | 79.6% |
| llm-wide-11008 | 255.01→250.59 | -1.7% | 258.55→254.15 | -1.7% | 98.6% |
| llm-deep-11008 | 258.86→251.40 | -2.9% | 260.20→255.84 | -1.7% | 98.3% |

**Geomean custom/cuBLAS**: 94.3% on the 13 "documented" shapes (94.2% on
2026-08-21 — unchanged within noise), 89.6% across all 26 (the added set
specifically targets harder occupancy/wave-quantization diagnostics, so a
lower geomean there is expected, not a regression).

**No real regressions.** Most shapes show a uniform ~1-3% dip in *both*
`custom` and `cuBLAS`/`cuDNN` — since the vendor libraries moved by
essentially the same amount as the custom kernel on the same shapes, this is
environment/clock-state noise (consistent with [[gpu_benchmark_clock_noise]]:
this A100 has no locked clocks and idles at 210MHz), not anything specific to
the kernel or the pending `ops_nv.py` diff. `custom/cuBLAS` ratios are stable
shape-for-shape.

## 1024³ fix, applied after this sweep was collected

Investigating the 1024³ gap (see `BF16_GEMM_NOTES.md`'s "1024³ bimodality"
entry) found that the already-existing four-warp `BLOCK_M=128` ILP kernel
(built for thin-N, `_ilp_128_source`) had never been tried on other
occupancy-limited shapes. Direct test with `GEMM_BLOCK_M=128` forced:

| shape | default (256-tile) | forced 128-tile | change |
|---|---:|---:|---:|
| 1024³ | 65.34 | 86.96 (repeat: 67-87 bimodal range, avg ~76) | **+20-33%** |
| thin-m-256 | 167.08 | 114.20 | **-32% (regression)** |

The difference is wave quantization: 1024³ has 32 CTAs by default, 64 with
the 128-tile (both comfortably under the 108-SM count, so halving the tile
is a clean win). thin-M=256 has 64 CTAs by default, 128 with the 128-tile —
crossing just above 108 SMs costs a mostly-wasted second wave, exactly the
`cta-boundary-112` cliff already documented in `BF16_GEMM_SWEEP_2026-08-21.md`.

Correctness verified two ways: the printed output sample is bit-identical
between the 256-tile and 128-tile paths for the same input, and a full-tensor
diff against an FP32 reference (4 shapes/seeds, including 1024³ twice) shows
only ordinary BF16 accumulation error (max_abs_diff 0.12-0.50 against
values with std ~90 for K=8192, no NaN/Inf) — not the garbage-value
signature of the uninitialized-shared-memory bugs from the two failed
small-tile attempts documented in `BF16_GEMM_NOTES.md`.

**Wired in** as an automatic dispatch rule in `_effective_block_m` /
`_effective_stages` (`nv_bf16_gemm.py`): the 128-tile path now also
auto-selects whenever halving the M-tile keeps the whole grid within one SM
wave (`(M//128)*(N//BLOCK_N) <= SM_COUNT`), in addition to the existing
thin-N (`N == BLOCK_N`) case. Checked against all 26 sweep shapes'
tile-count arithmetic: this is the **only** shape in the whole set whose
dispatch changes — every other shape's tile count is either already handled
by the persistent path (K≤512) or comfortably above the one-wave threshold,
so this is a targeted, zero-blast-radius change. Re-verified on hardware
with no explicit override needed: 1024³ → 6-stage ILP path automatically
(was 4-stage); square-2048, thin-m-256, thin-n-128, thin-k-128 all
unchanged, matching this sweep's numbers exactly.

1024³ moves from 76% of cuBLAS (65.49/86.47) to **~100% in its fast bucket,
~78% in its slow bucket** (86.96 or 67-78 depending on the still-unexplained
bimodality) — both buckets improved by the same ~20 TFLOPS, so the
bimodality mechanism itself is untouched, but the shape is faster either way.
