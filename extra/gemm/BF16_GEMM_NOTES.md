# NV BF16 GEMM: benchmark findings and tuning notes

Custom hand-written BF16 GEMM kernel for NV (A100), compared against cuBLAS
(`cublasGemmEx`) and cuDNN (graph matmul), across a range of shapes. This
documents what was measured, what was fixed, and what remains open with a
root cause, so future work doesn't re-derive it from scratch.

Kernel: `max_kernels/nv.bf16_fp32_bf16.3_stage.cu`, templated by
`nv_bf16_gemm.py` (BLOCK_M 128/256, BLOCK_K 32/64, STAGES 3-6, GROUP_M,
N_FIRST, CP_ASYNC mode, register-usage overrides).

## Measurement gotcha: this GPU's clock state is not stable by default

This A100 idles at 210MHz and only reaches its 1410MHz boost clock under
sustained load (`nvidia-smi -q -d CLOCK`; persistence mode is "Not Active").
Locking clocks (`nvidia-smi -pm 1` / `-lgc`) requires root, which is not
available here (no passwordless sudo), and is a shared-machine setting change
anyway, so don't reach for it lightly.

**Consequence**: with the benchmark script's default `WARMUP=500`, small/fast
shapes (any shape where 500 iterations don't add up to much wall-clock time)
give unstable, bimodal results — the same config can read e.g. 51 TFLOPS then
66 TFLOPS then 51 TFLOPS again on immediate reruns, with zero code changes.
Confirmed cause: clock ramp-up hasn't completed within the warmup window.

**Fix**: for shapes where a single iteration takes well under ~1ms, use a much
larger `WARMUP` (3000-5000) so total warmup wall-clock time is long enough for
the clock to actually reach boost and stay there. Large/deep-K shapes (K in
the thousands) don't need this — their per-iteration time alone is enough.
When in doubt, rerun 2-3x and check the number is stable before trusting it.

## Stable benchmark results (M, K, N; A@B, A is MxK, B is KxN)

All custom-kernel numbers use the default templated config unless noted.
TFLOPS, higher is better. "warmup" column: WARMUP used to get a stable read.

| M | K | N | custom | cuBLAS | cuDNN | custom/cuBLAS | warmup |
|---|---|---|---|---|---|---|---|
| 4096 | 4096 | 4096 | 252.5 | 252.3 | 251.2 | **win** | 500 |
| 8192 | 8192 | 8192 | 268.8 | 270.6 | 268.2 | 99% | 500 |
| 4096 | 4096 | 14336 | 257.5 | 253.5 | 261.1 | **win** vs cuBLAS | 500 |
| 4096 | 14336 | 4096 | 261.6 | 256.3 | 264.2 | **win** vs cuBLAS | 500 |
| 8192 | 8192 | 28672 | 264.2 | 266.9 | 266.1 | 99% | 500 |
| 8192 | 28672 | 8192 | 267.6 | 260.0 | 261.1 | **win** | 500 |
| 2048 | 8192 | 8192 | ~256 | 262.0 | 260.4 | 98% | 2000 |
| 8192 | 8192 | 2048 | ~258 | 261.2 | 260.8 | 98% | 2000 |
| 2048 | 2048 | 2048 | 154.4 | 159.3 | 159.2 | 97% | 5000 |
| 1024 | 1024 | 1024 | 51.4 | 86.0 | 76.9 | **60%** | 5000 |
| 8192 | **128** | 8192 (thin K) | 84.5 | 100.4 | 100.3 | **84%** | 5000 |
| 8192 | 8192 | **128** (thin N) | 85.2 (98.4 w/ `BLOCK_M=128`) | 123.7 | 124.3 | **69%** (79% tuned) | 5000 |
| **256** | 8192 | 8192 (thin M) | 172.4 | 205.9 | 205.2 | **84%** | 5000 |

**Bottom line**: the custom kernel wins or ties cuBLAS on every large/deep-K
production-shaped GEMM tested. It loses meaningfully on small/degenerate
shapes: a single-wave square (1024³), and shapes where one dimension is a
single tile or a shallow K (128).

## Fixes applied

1. **`GROUP_M` heuristic** (`nv_bf16_gemm.py`): was `2 if M <= 4096 else 8`,
   changed to `2 if M < 4096 else 8`. At M=4096 with a wide or deep other
   dimension (the 14336 shapes above), GROUP_M=8 gives +13%/+7% with zero
   regression on the M=4096 square case (verified: 208.3 vs 208.4 TFLOPS,
   noise-level). This is why those two rows show a win now.

2. **`GEMM_BLOCK_M=128` for thin-N (N=128) shapes**: verified, stable,
   reproducible +15% (85.2 → 98.4 TFLOPS) via a register-count sweep. Not
   wired in as an automatic default — `BLOCK_M`/`SHARED_MEM` are module-level
   constants computed once at import in `nv_bf16_gemm.py`, so making this
   shape-adaptive needs threading BLOCK_M through the templating functions as
   a parameter instead of a global. Worth doing if someone revisits this file.
   Note `BLOCK_M=128` is *not* a general win — it regresses every other shape
   tried (square 1024³, thin-M=256, deep-K=128 all got worse), because the
   existing 128-path (`_block_m_source`) halves the M-fragments per warp
   without compensating, losing instruction-level parallelism. Only use it
   when N is the single-tile bottleneck.
   **Superseded for 1024³ specifically** by the four-warp ILP path added
   later (see "Small-tile follow-up" and the "1024³ fix" section below) —
   this paragraph describes the original 8-warp `_block_m_source` path's
   behavior, which is a different code path from the four-warp
   `_ilp_128_source` one. thin-M=256 and deep-K=128 still regress under
   either variant.

## Open gaps, with root cause (not fixed — see "what was tried" below)

### Thin-N=128, thin-M=256, square 1024³: CTA-count-limited occupancy

These shapes launch far fewer CTAs than the GPU has SMs (108) under the
standard 256×128 tile (e.g. N=128 → only 32 CTAs). **Registers, not shared
memory, are the actual occupancy limiter**: `ptxas -v` shows both the
original kernel (230 regs/thread) and a shrunk-shared-memory variant (216
regs/thread) need ≤128 to fit 2 CTAs/SM at 256 threads/block (256×128 =
32768, half of the SM's 65536-register file); neither gets close. A real fix
needs a smaller output tile *with* a rebalanced warp layout that preserves
per-warp instruction-level parallelism (not just fewer M-fragments, which is
what the existing 128-path does and why it mostly regresses) — that's a new
kernel design, not a knob.

### Shallow K=128 at large M,N: wave-transition overhead

At M=N=8192, K=128 launches the same 2048 CTAs as the deep-K case (only K
changed), giving ~19 sequential waves per SM (2048 CTAs / 108 SMs, 1 CTA/SM
due to the register limit above). Static SASS inspection
(`cuobjdump --dump-sass`) confirmed the 4-iteration K-loop is **fully
unrolled** (one `BRA`, and it's the kernel-exit trap, not a loop branch) — so
there's no loop-control overhead to blame. The real signal: **the custom
kernel already beats cuBLAS on this exact K=128 shape at low wave counts**:

| M | N | K | waves | custom | cuBLAS | result |
|---|---|---|---|---|---|---|
| 1024 | 8192 | 128 | ~2.4 | 61.2 | 54.3 | **custom wins +13%** |
| 2048 | 8192 | 128 | ~4.7 | 73.8 | 66.4 | **custom wins +11%** |
| 8192 | 8192 | 128 | ~19 | 84.5 | 100.4 | custom loses -16% |

So the per-wave transition cost compounds faster for the custom kernel than
for cuBLAS as wave count grows. The mechanically correct fix is a
**persistent kernel** (launch ~108 CTAs, each looping internally over its
share of the 2048 tiles, avoiding repeated CTA-launch overhead).

**What was tried and failed**: wrapping the existing kernel's body in a
grid-stride loop (`for (pid = blockIdx.x; pid < total_tiles; pid +=
NUM_CTAS)`), reusing all of its proven swizzle/warp/epilogue math unchanged.
This surfaced a real tinygrad bug (see below) and, once fixed, turned out not
to help: even at exact 1:1 CTA-to-tile mapping (the loop body runs exactly
once per CTA, identical register count to the original per `ptxas -v`),
performance dropped to ~33 TFLOPS from the original's 84.5. **The loop
construct itself costs ~2.5x, independent of any wave-amortization benefit**
— the compiler can no longer schedule/optimize the body as aggressively once
it's inside a runtime loop instead of being straight-line/fully-unrolled
code. Closing this gap for real needs a kernel written to be loop-friendly
from the ground up, not an existing unrolled kernel adapted into one — a
materially bigger rewrite than anything else attempted here.

A separate single-buffer (STAGES=1-equivalent) shallow-K kernel was also
tried, on the theory that a smaller shared-memory footprint would allow 2+
CTAs/SM. That was wrong too, for the same register-limited-occupancy reason
above (24KB shared memory doesn't help when registers already cap you at 1
CTA/SM), and it also threw away the original's load/compute overlap, landing
at 76 TFLOPS — worse than baseline. Discarded.

## Bug found in tinygrad itself: `gridDim.x` reads as 0 on the NV backend

While building the persistent-kernel experiment above, a grid-stride loop
using `pid += gridDim.x` hung the GPU (`Wait timeout`, needs device recovery)
even in cases with no real looping needed. Root cause, confirmed with a
minimal probe kernel (writes `gridDim.x` to an output buffer):

```
global_size passed to launch: (4, 1, 1)
gridDim.x read inside the kernel: 0
```

**`gridDim` reads as zero inside kernels launched through tinygrad's NV
driver-direct backend** (`tinygrad/runtime/ops_nv.py`), regardless of the
actual launch grid size. This backend submits directly via ioctl/HCQ rather
than going through `cuLaunchKernel`, and apparently doesn't populate whatever
constant-bank location the `%nctaid` PTX register read comes from. Any custom
kernel relying on `gridDim` for a grid-stride loop will get `pid +=
0` and loop forever (or, if the loop is written the other way, silently do
the wrong thing without hanging). The workaround used here: bake the intended
CTA count as a compile-time `#define` and use that instead of reading
`gridDim.x` at runtime. This should be reported/fixed upstream in
`ops_nv.py` for anyone else writing custom kernels against this backend.

## Update: raw-PTX pipeline rewrite (applied to the real kernel file)

Follow-up session, continuing past the "land here" recommendation below (kept
for history). Root-caused why the persistent-kernel experiment looked like a
dead end, found it wasn't, and landed one more real fix.

### The "loop costs 2.5x" conclusion above was wrong — it was a bug, not a limit

Re-diffed the SASS between the flat baseline and the loop-wrapped experiment
and found the real cause: the loop-wrapped version was built directly from
the **raw** `.cu` file text (for convenience), which has the naive "slower
way" per-element epilogue as its literal committed content — the efficient
shared-memory-staged, 128-bit-vectorized epilogue only exists because
`nv_bf16_gemm.py`'s `_shared_epilogue_source()` rewrites it in as a
post-processing step, which the experiment never applied. So the "persistent
kernel" comparison was efficient-epilogue baseline vs. naive-epilogue
experiment — not a fair test. Confirmed via `cuobjdump --dump-sass`: the
experiment's epilogue was 128× `STG.E.U16` (one bf16 element per store)
instead of the baseline's `STS`+`LDS.128`+`STG.E.128` (staged, coalesced
128-bit stores). Once the shared-epilogue transform was correctly reapplied
on top of the loop-wrapped body, the persistent kernel actually **beat** the
flat baseline (89.5 vs ~82-84 TFLOPS average on the K=128/M=N=8192 shape) —
the wave-amortization theory was right all along; the implementation had a
missing optimization, not the loop itself being the problem.

### Real, applied fix: raw PTX `cp.async` instead of the `<cuda_pipeline.h>` intrinsics

While isolating the epilogue bug, also found (and confirmed via repeated,
contemporaneous A/B comparison, not a one-off measurement) that hand-writing
`__pipeline_memcpy_async` / `__pipeline_commit` / `__pipeline_wait_prior` as
raw `asm volatile` blocks — instead of using `<cuda_pipeline.h>`'s intrinsics
— is faster, with no regressions found across every shape and config tested
(all `STAGES` 3-6, `BLOCK_M` 128/256, `BLOCK_K` 32/64, all `CP_ASYNC` modes).
Mechanism isn't fully understood (the emitted PTX instructions are
byte-for-byte the same — verified with `nvcc -ptx`/`ptxas -v` — so the
intrinsics' implicit pipeline-object bookkeeping must be constraining the
compiler's scheduling somehow), but the effect is real and reproducible.
Sample gains (all re-verified 2-3x before trusting):

| shape | before | after | note |
|---|---|---|---|
| 8192³ | 268.8 | 269.6-270.0 | +0.3-0.4% |
| 4096³ | 252.5 | 253-257 | +0.5-2% |
| 1024³ | 51.4 | **usually ~65 (+27%), occasionally still ~51-55** | see caveat below |
| K=128, M=N=8192, `GROUP_M=0` | 84.0 | 87.6-87.75 | +3.7%, see GROUP_M note below |

**Applied to the real kernel file** (`nv.bf16_fp32_bf16.3_stage.cu`): the
three intrinsics are now hand-defined with the same call-site names (so
`nv_bf16_gemm.py`'s text-templating needs no changes), using
`cp.async.cg.shared.global` by default (matches the intrinsics' previous
default behavior — confirmed via `-ptx` dump). `GEMM_CP_ASYNC` default
changed from `"ca"` to `"cg"` to match (the old `"ca"` name never actually
selected `.ca` mode before this change — it was a no-op "leave the intrinsic
default alone" that happened to already be `.cg`; explicit `"ca"`/`"cg128"`
now genuinely patch the mnemonic). Needed one incidental fix: removing
`#include <cuda_pipeline.h>` also removed the transitive `uint32_t`
definition the ldmatrix helpers rely on — added `typedef unsigned int
uint32_t;` since NVRTC has no `<cstdint>`.

**Caveat on 1024³**: repeated fresh-process measurements show real
bimodality — mostly ~65 TFLOPS, occasionally dropping to ~50-55 with no code
or config change (9 samples: 4× ~50.7, 1× ~55.4, 4× 65.19). The *old* kernel
never showed this bimodality across many measurements this session (always a
tight 51.3-51.4). Hypothesis, not confirmed: possibly a memory-allocator- or
address-dependent DRAM/L2 partition-camping effect that the new kernel's
different register/instruction layout made newly sensitive to. Don't quote a
single "+27%" for this shape without rerunning it a few times first.

### New finding: `GROUP_M`'s default interacts badly with shallow-K, wide-M/N shapes

The K=128 win above only shows cleanly under `GEMM_GROUP_M=0`. At the real
default (`GROUP_M=8` for M≥4096), K=128 still measures ~84.5 — unchanged from
before this fix — because `GROUP_M=8`'s grid-swizzling costs about as much as
the raw-PTX change gains, on this specific shape. Conversely `GROUP_M=0` on
the *square* 8192³ shape costs ~4% (258 vs 269-270 with `GROUP_M=8`). So the
current single M-based threshold (`2 if M < 4096 else 8`) isn't the whole
story — it should probably also depend on K (or on total tile count vs SM
count). Not fixed; a real next step if pursued.

### Persistent kernel: integrated, gated on K depth (not tile count)

With the epilogue bug fixed, re-investigated what actually predicts whether
the persistent grid-stride-loop kernel (108 CTAs, each looping over its
share of the output tiles; see "Bug found in tinygrad itself" below for the
`gridDim.x` workaround it needs) helps or hurts. The initial hypothesis
("helps when total tile count ≫ SM count") was wrong — the real predictor is
**K depth, independent of tile count**:

| M | N | K | tiles | flat | persistent (108 CTAs) |
|---|---|---|---|---|---|
| 8192 | 8192 | 128 | 2048 | 87.6 | **89.5** |
| 8192 | 8192 | 256 | 2048 | 109.4 | **140.3** (+28%) |
| 8192 | 8192 | 512 | 2048 | 183.2 | 183.6 (tie) |
| 8192 | 8192 | 1024 | 2048 | 221.6 | 216.9 (-2%, regresses) |
| 8192 | 8192 | 8192 | 2048 | 269.5 | 235.9 (-12%, regresses) |
| 4096 | 4096 | 128 | 512 | 58.2 | **62.3** (+7%) |
| 2048 | 2048 | 128 | 128 | 36.9 | **40.0** (+8.5%) |
| 8192 | 128 | 128 | 32 | 17.0 | **19.9** (+17%) |
| 8192 | 128 | 8192 | 32 | 84.7 | 74.8 (-12%, regresses) |
| 256 | 8192 | 8192 | 64 | 171.9 | 150.0 (-13%, regresses) |

At **every** tested tile count (32 up to 2048), shallow K (≤512) wins and
deep K (≥1024) loses. Root cause: for shallow K the inner K-loop fully
unrolls (confirmed via SASS — a single `BRA`, the kernel-exit trap, not a
real branch), so the outer persistent loop is the only real loop in the
kernel and compiles fine; for deep K the inner K-loop is itself a real loop
(too many iterations to unroll), and nesting a real loop inside another real
loop compiles measurably worse than either alone. The earlier "regression"
readings for N=128 and M=256 above were confounded — those tests happened to
use K=8192 (deep), not because they had few tiles.

**Integrated**: `_custom_nv_bf16_gemm` now dispatches to the persistent path
when `K <= GEMM_PERSISTENT_K_THRESHOLD` (default 512) and the config is the
validated one (`BLOCK_M=256`, `BLOCK_K=32`; `GEMM_PERSISTENT=0/1` forces
off/on). `GROUP_M` grid-swizzling and `SERPENTINE` mma-reordering are both
skipped on the persistent path (untested combination; the persistent path
uses plain linear tile decomposition instead). `GEMM_SM_COUNT` (default 108)
controls the number of persistent CTAs — must match the actual GPU's SM
count to get the benefit; being off by ~25% loses most of it. Verified: every
non-dispatched shape (K > threshold) reads identical to before the change;
every dispatched shape improves or ties, none regress.

Net effect on the original three-shapes table: **thin K (8192,128,8192) goes
from 85% of cuBLAS to ~88-90%.** Thin N (8192,8192,**128**) and thin M
(**256**,8192,8192) are unaffected by this fix specifically because those
rows use K=8192 (deep) — their bottleneck is occupancy (too few tiles versus
108 SMs), not K depth, and needs the different fix below.

## Recommendation (superseded, kept for history — see below)

~~Land here: the custom kernel already wins or ties cuBLAS on every
large/production-shaped GEMM tested, with two verified fixes applied
(`GROUP_M` heuristic, documented `BLOCK_M=128` knob for thin-N). The
remaining gaps (thin-M/N single-tile shapes, shallow-K at high wave count)
are real, root-caused, and would need new kernel designs (not tuning) to
close — a rebalanced small-tile warp layout for the occupancy-limited shapes,
and a from-scratch loop-native design for the wave-transition-limited shapes.
Both are multi-day efforts with real execution risk (as this session's three
failed attempts show) rather than incremental tuning.~~

This turned out to be too pessimistic — see "Update: raw-PTX pipeline
rewrite" above. The persistent-kernel design wasn't a dead end, it had a bug
in the experiment (missing epilogue optimization), and a separate, real,
broadly-applicable fix (raw-PTX pipeline intrinsics) came out of debugging
it.

## Final result table (post raw-PTX fix, real defaults, fresh contemporaneous run)

| M | K | N | custom | cuBLAS | cuDNN | custom/cuBLAS |
|---|---|---|---|---|---|---|
| 1024 | 1024 | 1024 | 65.2 | 86.2 | 74.0 | 76% (was 60%; see bimodality caveat above) |
| 2048 | 2048 | 2048 | 152.9 | 159.1 | 159.3 | 96% |
| 4096 | 4096 | 4096 | 252.9 | 255.0 | 245.1 | 99% (beats cuDNN) |
| 8192 | 8192 | 8192 | 269.5 | 270.7 | 269.5 | **>99.9%, ties cuDNN** |
| 4096 | 4096 | 14336 | 253.2 | 258.0 | 258.1 | 98% |
| 4096 | 14336 | 4096 | 258.8 | 260.8 | 261.0 | 99% |
| 8192 | 8192 | 28672 | 260.8 | 266.3 | 266.1 | 98% |
| 8192 | 28672 | 8192 | 264.1 | 259.9 | 259.9 | **win**, +1.6% |
| 2048 | 8192 | 8192 | 247.1 | 257.5 | 253.5 | 96% |
| 8192 | 8192 | 2048 | 247.0 | 259.9 | 253.5 | 95% |
| 8192 | **128** | 8192 (thin K) | 84.5 | 99.9 | 100.0 | 85% |
| 8192 | 8192 | **128** (thin N) | 84.7 | 124.1 | 124.5 | 68% |
| **256** | 8192 | 8192 (thin M) | 168.5 | 206.1 | 205.4 | 82% |

Every large/production shape sits at 95-100%+ of cuBLAS, with one outright
win. The three still-open degenerate shapes (thin K, thin N, thin M) are the
ones needing the shape-conditional persistent-kernel dispatch and/or a new
small-tile design described above.

## Split-K: implemented, correct, but doesn't pay off in practice (yet)

Attempted this as the fix for the occupancy-limited shapes (thin N=128, thin
M=256, 1024³): split K across `GEMM_SPLIT_K` independent kernel launches
(each a normal, full-tile invocation over a K-slice, addressed via a
compile-time loop-bound override and address offset -- see
`_split_k_source` -- no buffer copies, reuses the existing tile/warp/swizzle
code unchanged), combined via `Tensor.stack(...).cast(float32).sum(axis=0)`.
Two real bugs surfaced and got fixed along the way (both now covered by the
correctness checks in the test harness that caught them):

- `_persistent_source` wholesale-replaces its header region, which silently
  discarded `_split_k_source`'s `num_k_blocks` override whenever the two were
  combined -- gave `max_abs_diff=inf` (not just imprecise, actually wrong).
  Fixed by giving `_persistent_source` an explicit `num_k_blocks_expr` param.
- Naive Python-loop benchmarking without `TinyJit` measured 13-20 TFLOPS due
  to per-call graph-tracing overhead having nothing to do with the kernel --
  not a real regression, just a broken benchmark methodology. Always use
  `TinyJit` (or the low-level hw-queue timestamp approach) for these numbers.

**What actually happens, measured three ways** (raw GPU-compute-only via hw
queue timestamps, no reduction kernel; full-pipeline via `TinyJit`, which
does include the reduction; and repeated to check for the address-dependent
noise this session has run into repeatedly on small-output shapes):

| shape | split | raw GPU compute time | full pipeline (TinyJit, incl. reduction) |
|---|---|---|---|
| N=128, M=K=8192 | off | ~65-70 | ~65-75 |
| N=128, M=K=8192 | 2 | **~80** (+~20%) | ~61-66 (**net loss**) |
| M=256, N=K=8192 | off | ~150-169 | — |
| M=256, N=K=8192 | 2 or 4 | lower at every factor tried | — |
| 1024³ | off | 51.7 | — |
| 1024³ | 2 or 4 | lower at every factor tried (40.9, 30.1) | — |

Only N=128 shows a real GPU-compute-time gain from splitting, and even there
it's **erased once the required reduction kernel and second launch's
dispatch overhead are included** — the raw-compute-only measurement was
misleadingly optimistic because it skipped the actual combination step.
Root cause for why it doesn't generalize:

- **M=256 and 1024³ never show a compute-time gain at all.** M=256 already
  has 64 tiles (59% of the 108 SMs) -- splitting to 128 tiles overshoots 108
  and reintroduces wave-transition cost, canceling the parallelism gain.
  1024³ is fundamentally tiny in absolute terms (2.1 GFLOP, ~0.04ms total)
  -- splitting multiplies the fixed per-launch prologue/epilogue cost while
  each piece does proportionally less useful work, a straightforward net
  loss when the *absolute* per-piece time is already small. (1024³ appears
  to be near an inherent floor for this kernel design at this problem size,
  not a fixable gap.)
- **N=128's gain requires the split pieces to stay deep enough that
  fixed-cost-per-launch doesn't dominate** (each piece was K=4096, still
  substantial), which is also exactly why it stops being a net win once a
  second real kernel launch (the reduction) is added to the critical path.

To actually realize split-K's benefit would need the combination step to be
near-free -- e.g. atomic accumulation directly in the GEMM kernel's epilogue
into a shared FP32 buffer, avoiding a second kernel launch and buffer
allocation entirely -- which was not attempted (real added complexity:
atomics correctness, a separate FP32 temp buffer, zeroing it before use).
`GEMM_SPLIT_K` is implemented, tested correct, and left in the codebase
(off by default) since it's genuinely a working, documented, honest
negative result rather than a half-finished feature -- future work on
occupancy-limited shapes should start from "the stack+cast+sum combination
step is the actual bottleneck" rather than re-deriving this.

## Small-tile ILP-preserving kernel: attempted, not achieved

Researched two external kernel repos for techniques (user-suggested:
sonnyli/flash_attention_from_scratch, gau-nernst/gn-kernels) before
attempting this by hand again. Useful finding, worth keeping in mind for any
future attempt: professional/templated kernels (gn-kernels' `swizzle<STRIDE>`
in `common.h`, matching CUTLASS's own canonical `(B,M,S)` swizzle atom) keep
the XOR swizzle a pure function of **row pitch in bytes**
(`BLOCK_K * sizeof(dtype)`), with zero dependence on `BLOCK_M`/`BLOCK_N`/warp
count — in a *cleanly parameterized* kernel, shrinking the output tile
shouldn't require re-deriving the swizzle at all, only the warp-to-fragment
index arithmetic on top of it.

That finding doesn't transfer cleanly to *this* kernel, though. Traced
through why: `nv.bf16_fp32_bf16.3_stage.cu`'s swizzle phase term
(`(load_smem_a_phase + {0,2,4,6}) ^ (threads % 8)`) isn't just a
bank-conflict rotation independent of the fragment layout — the `+0/+2/+4/+6`
increments are simultaneously *selecting which of 4 fragments* (M-subgroup ×
K-half) a given ldmatrix call reads, fused into the same expression as the
conflict-avoidance XOR. Unlike gn-kernels' cleanly separated
`swizzle(row, col)` (a pure function you call with whatever row/col you want)
and `A_rmem[WARP_M/MMA_M][BLOCK_K/MMA_K][4]` (fragment indexing kept
separate), this kernel's swizzle and fragment-selection are the same
expression — so "just change which fragments a warp covers" isn't
separable from "re-verify the swizzle," contrary to what the research
suggested for well-templated kernels.

Two attempts, both hit real correctness bugs:

1. **From-scratch unswizzled small tile** (128×128, `tz=1` so every warp
   covers the full 128-wide N with 2 M-fragments × 16 N-fragments = 32
   MMA/warp, matching the original's per-warp ILP): compiled and ran without
   hanging, but gave wrong output. Debugged with `B = identity` (so `C`
   should exactly equal `A`, making any addressing bug immediately visible
   as a row/col permutation) — the actual mismatches weren't a clean
   permutation, they were denormals and huge-exponent garbage values
   (`-3e-32`, `3e37`, etc.), meaning the load formula was reading
   **uninitialized shared memory**, not just the wrong row/col. That's a
   structural bug in the store/load coverage, not a small indexing offset.
2. **Reuse the proven swizzle, only touch fragment-to-row assignment**
   (keep `nv.bf16_fp32_bf16.3_stage.cu`'s exact A/B swizzle formulas, just
   use 2 `wg_m` groups instead of 4 while keeping 4 fragments/warp, to shrink
   `BLOCK_M` 256→128 without reducing ILP): got as far as realizing the
   `+0/+64meta/phase+4/phase+6` fragment offsets aren't simple row jumps —
   they fuse M-subgroup selection into the same swizzle phase used for
   bank-conflict avoidance (see above) — before finding a safe way to adapt
   them. Stopped before writing/testing a full kernel, since the risk of
   another silent-wrong-answer bug was high without being able to fully
   trace the phase encoding by hand.

Both attempts were deleted rather than left in the tree in a broken state.
**Assessment stands from earlier in this document: this needs either a
profiler to verify a hand-derived swizzle/fragment mapping, or rewriting the
kernel from scratch with a gn-kernels-style clean separation between the
swizzle function and the fragment index math** (i.e. not adapting
`nv.bf16_fp32_bf16.3_stage.cu`'s fused expressions, but redesigning them to
be separable first) — a materially larger undertaking than adapting the
existing kernel, closer to "write a new kernel family" than "fix a tile
size."

## Small-tile follow-up: achieved without re-deriving the A swizzle

The assessment above missed a simpler composition of two already-proven
pieces. The existing `BLOCK_M=128` transform already has a correct
unswizzled A layout; its performance problem was only that the original
eight-warp CTA left each warp with 2x8 MMA fragments. The working variant
uses four warps (`tz=1`), retains that A layout unchanged, and gives each
warp both of the old `wg_n` B-fragment groups. The second group is the same
proven B swizzle at phase offsets `+2/+6/+10/+14`, so no new swizzle was
derived. This restores 2x16 MMA fragments per warp.

The garbage-output failure in the first attempt had a concrete structural
cause: after halving the thread count, its async-copy schedule still assumed
256 threads. It initialized only A rows 0:32 and 64:96, and B rows 0:8 and
16:24. The working schedule issues four A and four B copies per lane, covering
all shared-memory rows before any `ldmatrix` reads them.

Correctness was checked with an exact BF16 identity GEMM and random multi-CTA
shapes; all outputs were finite and the identity result was bit-exact. The
final path also stages its epilogue in shared memory and uses 128-bit global
stores, matching the coalescing strategy of the 256-row kernel. `ptxas -v`
reports 254 registers/thread and no spills. Contemporaneous A/B results on the
deep-K thin-N target (`M=K=8192, N=128`, 3000 warmups):

| path | TFLOPS | time |
|---|---:|---:|
| old 128x128, 8 warps | 98.35 | 0.175 ms |
| four warps/full N, scalar epilogue, 4 stages | 110.87-110.93 | 0.155 ms |
| coalesced epilogue, 4 stages | 113.89-113.91 | 0.151 ms |
| coalesced epilogue, 6 stages | **115.80-115.99** | **0.148 ms** |
| cuBLAS GemmEx | 124.57 | 0.138 ms |

That is a reproducible **+17.9%** over the previous tuned path, moving the
custom kernel from 79% to 93% of the fresh cuBLAS result. Deep-K `N=128`
selects this path and six pipeline stages automatically when neither
`GEMM_BLOCK_M` nor `GEMM_STAGES` is explicitly set; shallow K retains the
existing persistent 256x128/four-stage path. `GEMM_ILP_128=0` keeps the old
eight-warp 128x128 implementation for A/B comparisons.

## 1024³ fix (2026-08-24): generalized the four-warp 128-tile dispatch

The four-warp `BLOCK_M=128` ILP path (see "Small-tile follow-up" above) was
only ever auto-dispatched for thin-N (`N == BLOCK_N`). Testing it on the
other occupancy-limited shapes found it's also a real win for 1024³
(+20-33%, 65.34→86.96 TFLOPS, moving it to ~100% of cuBLAS in its fast
bucket) but a real regression for thin-M=256 (167.08→114.20, -32%). The
difference: 1024³'s CTA count goes 32→64 (both under the 108-SM count,
clean win), while thin-M=256's goes 64→128 (crosses just above 108 SMs,
triggering the same wave-quantization cliff as `cta-boundary-112`).

Generalized `_effective_block_m`/`_effective_stages` (now take `(m, n,
k_depth)`) to auto-select the 128-tile path whenever `(M//128)*(N//BLOCK_N)
<= SM_COUNT`, in addition to the existing thin-N case. Checked against
every shape in `sweep_bf16_gemm.py`: 1024³ is the **only** one this changes
— everything else is either K≤512 (handled by the persistent path
already) or has a tile count comfortably above the threshold either way.
Verified on hardware with no env override needed. Correctness checked via
bit-identical output samples across tile sizes plus a full-tensor FP32-
reference diff (see `BF16_GEMM_SWEEP_2026-08-24.md` for the numbers) — clean
BF16 rounding error, not the garbage-value signature the two failed
small-tile attempts above hit.

This closes 1024³. **thin-M=256 is still open** — the 128-tile path doesn't
transfer, so it still needs either a smaller/different tile shape that
doesn't cross the wave boundary, or the cheaper split-K reduction described
below.

## Current recommendation

Applied this round: raw-PTX pipeline rewrite (broad win, no regressions),
shape-conditional persistent-kernel dispatch (fixes thin-K specifically),
split-K (implemented, correct, but not a net win in practice), the
four-warp 128x128 ILP path (improves deep-K thin-N by 17.9%), and its
generalization to 1024³ (+20-33%, see above). Still open, in priority order
if continuing:

1. **thin-M=256 remains occupancy-limited** — confirmed the 128-tile ILP
   fix that closed 1024³ does not transfer (regresses it, wave-quantization
   cliff at 128 CTAs). Still needs a genuinely different output-tile shape
   that doesn't cross the 108-SM boundary, or a cheaper split-K combination
   step. Atomic FP32 accumulation in the GEMM epilogue remains one possible
   way to eliminate the separate `stack/cast/sum` reduction, with the
   temp-buffer lifecycle and zeroing complexity described above.
2. ~~`GROUP_M` heuristic vs. shallow-K tension~~ — **checked 2026-08-24,
   turns out not to be an issue.** Direct A/B on the flat-dispatch shapes
   just above the persistent threshold (K=768, K=1024 at M=N=8192, real
   hardware, contemporaneous): `GEMM_GROUP_M=0` vs. the real default
   (`GROUP_M=8` since M≥4096) are a tie at K=768 (200.05 vs 200.50 TFLOPS)
   and the *default is actually better* at K=1024 (216.31 vs 210.09, default
   +3%) — the opposite of the ~4% cost this item worried about. The original
   concern was measured on the *pre-persistent-dispatch* K=128 case, where
   K=128 still went through the flat grid-swizzled kernel; now that K≤512
   always dispatches to the persistent path (`group_m=0` unconditionally,
   see `_custom_nv_bf16_gemm`), that regime doesn't hit this heuristic at
   all anymore. Nothing to fix here — closing this item.
3. **1024³ bimodality** — reproduced again 2026-08-24 on a fresh dedicated
   A100 allocation (no other tenants): 6 fresh-process runs gave a clean
   65/65/65/51/51/65 split, sample outputs bit-identical across all runs
   (same seed), confirming it's purely a *timing* artifact, not correctness.
   Checked the leading hypothesis directly: printed every group's A/B/C
   virtual addresses across multiple process runs — **they are fully
   deterministic** (same base, same sequential layout, identical across
   restarts), which rules out a *virtual*-address-dependent theory. Since
   `benchmark_custom`'s single timed region cycles through all 30 input
   groups every run (300 iters / 30 groups), the effect can't be "this run
   picked a slow buffer" — it has to be something process-global (most
   likely physical DRAM page placement or L2-slice mapping assigned by the
   driver at allocation time, which differs run-to-run even though the VA
   layout doesn't). Can't go further without hardware counters (Nsight
   Compute is blocked here: `ERR_NVGPUCTRPERM`). Treat as a real, understood-
   as-far-as-possible-without-a-profiler open item, not a regression to chase
   further with guess-and-check.

## Update 2026-08-24: `tinygrad/runtime/ops_nv.py` upstream changes

Separately from the kernel file itself, a pending (uncommitted) diff to
tinygrad's actual NV backend (`ops_nv.py`, plus small plumbing in
`device.py`/`uop/ops.py`/`ops_cuda.py`) adds:

- **`shared_mem` plumbing**: `KernelInfo(shared_mem=...)` (already used by
  `_custom_nv_bf16_gemm`, see `shared_mem = stages * (block_m*BLOCK_K +
  BLOCK_K*BLOCK_N) * dtypes.bfloat16.itemsize` above) now flows through
  `ProgramInfo` → `TinyELF` → `NVProgram`/`CUDAProgram` so the compiled
  program's actual dynamic shared-memory footprint reaches the QMD /
  `cuFuncSetAttribute`-equivalent path correctly, instead of relying only on
  what's discoverable from the compiled ELF's static `.nv.shared.<name>`
  section.
- **Ampere `AMPERE_COMPUTE_A` / QMD-version-2 support**: this specific A100
  (SLURM `sacramento`, A100-SXM4-40GB) reports `compute_class == 0xc6c0`
  (`AMPERE_COMPUTE_A`), which the old `ops_nv.py` never handled (only
  `AMPERE_COMPUTE_B` and newer). Confirmed live on real hardware
  (2026-08-24): `Device['NV']` now reports `qmd_version=2,
  compute_class=0xc6c0, arch=sm_80` and runs a real matmul correctly.
  **This looks like a genuine "the NV backend may not have worked on this
  exact A100 at all before this fix" bug**, not just cleanup — worth
  keeping in mind if older measurements in this doc need re-verifying,
  though nothing found so far contradicts them.
- A smarter shared-memory carveout calc (`_smem_config`) that targets fitting
  multiple CTAs per SM (up to GA100's 164 KiB carveout) instead of just the
  smallest config that fits one. **Verified this does not change the
  occupancy-limited shapes** (thin-M=256, thin-N=128, thin-K=128, 1024³, and
  square-8192 as a control all matched their documented baselines within
  normal noise on 2026-08-24) — consistent with the existing register-count
  root cause (230-254 regs/thread already caps 1 CTA/SM), confirming smem
  headroom alone doesn't help those shapes.
- GPU-minor/device-node accessibility filtering and a driver-version-query
  fallback, aimed at containerized/restricted-node environments where RM can
  report GPUs whose device nodes aren't actually openable.

Tests: `test/unit/test_ops_nv.py` (5 new unit tests, all passing) and one
new mockgpu test (`test_nv_device_with_remapped_minor`, passing); the one
failure in `test/testextra/test_mockgpu.py`
(`test_import_typing_extensions`) is a pre-existing circular-import issue on
unmodified master, unrelated to this diff.
