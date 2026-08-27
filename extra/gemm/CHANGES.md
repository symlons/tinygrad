# A100 NV backend + GEMM kernel — change summary

Four commits, in order. Each is independently motivated; later ones build on earlier ones.

## 1. `nv: Ampere GA100 (A100) QMD-v2 support`

**Problem:** the NV backend's launch descriptor (QMD) code only ever handled QMD major
version 3 (and 5 for Blackwell), hardcoding `qmd_major_version: 3` for every pre-Blackwell
GPU. This A100's `compute_class` (`AMPERE_COMPUTE_A`, GA100) actually requires **QMD
version 2** — a different field layout. Feeding hardware a mistagged/misformatted
descriptor broke device init and kernel execution on this GPU outright.

**Fix:** QMD-v2 field layout, v2's differently-named semaphore/release fields, and a
shared-memory carveout calc that targets fitting multiple CTAs per SM (GA100 supports
larger 132/164 KiB configs, not just the smallest config that fits one CTA). Also adds a
driver-version-query ioctl fallback for RM proxies that reject the normal
`NV0000_CTRL_CMD_SYSTEM_GET_BUILD_VERSION_V2` control call outright but still support the
lower-level ioctl libcuda itself falls back to.

## 2. `nv/cuda: dynamic shared-memory support (KernelInfo.dynamic_smem)`

**Problem:** A100 caps *static* `__shared__` at 48 KiB per block even though it has up to
164 KiB per SM — anything larger must be requested as *dynamic* shared memory
(`extern __shared__`), whose size lives outside the compiled kernel image and has to be
told to the launch descriptor separately. tinygrad's own generated kernels never need
this (they only ever reference `blockIdx`/`threadIdx`), so nothing plumbed it through.

**Fix:** `KernelInfo.dynamic_smem` → `ProgramInfo` → `TinyELF` → `NVProgram`/`CUDAProgram`,
landing in the QMD's `shared_memory_size` (NV) / `cuLaunchKernel`'s smem argument (CUDA).
Named `dynamic_smem`, not `shared_mem`: `NVProgram` *adds* it on top of whatever static
`.nv.shared` the compiled kernel already declares, so a name implying "total" would invite
double-counting a kernel that has both a static and a dynamic component. Verified with a
real `NVProgram` built under MOCKGPU — the value is checked against the actual QMD bytes,
not just that it survives the dataclass hops.

## 3. `extra/gemm: hand-tuned A100 BF16 GEMM kernel, competitive with cuBLAS/cuDNN`

Hand-tuned CUDA GEMM kernel (raw-PTX cp.async pipeline, shape-conditional persistent-kernel
dispatch, split-K) that wins or ties cuBLAS/cuDNN across the full shape range tested, not
just a benchmark-flattering square case. Needs commit 2's dynamic shared memory (up to
96 KiB for the default tile config) to run on A100 at all. Full investigation log,
benchmark tables, and dead-end write-ups: `BF16_GEMM_NOTES.md`.

## 4. `extra/gemm: launch-time GPU-selection script for restricted GPU environments`

**Problem:** commit 1 originally *also* added in-tinygrad probing of which `/dev/nvidiaN`
nodes are actually openable, because NVIDIA RM's own enumeration can report far more GPUs
than exist as device nodes in a restricted/proxied sandbox — confirmed live on Modal's
gVisor+nvproxy A100 containers: RM reported 16 "valid" GPUs against exactly 1 real device
node, with a *different* accessible minor number on every separate container launch (5/5
samples, no two the same). No environment variable set once could ever stay correct.

**Decision:** that probing was deliberately removed from tinygrad's NV backend entirely —
device selection stays plain upstream (`DEV=<idx>+NV` indexes RM's raw, unfiltered list).
Job-launch-time environment quirks don't belong in the library. `modal_select_gpu.sh`
does the discovery instead, outside tinygrad, at launch time: lists `/dev/nvidiaN` nodes
directly (no ioctls) and sets `DEV` before `exec`ing the real command. Confirmed working
end-to-end live on Modal.

**Known tradeoff, accepted deliberately:** the removed in-tinygrad approach reconciled by
matching actual `minor_number` values, correct regardless of gaps in RM's list. This
script instead assumes RM's list is gap-free/ascending (position == minor number) — true
in every sample observed, but not a documented driver guarantee. If that assumption ever
breaks on some platform, this fails silently (wrong GPU, or a confusing downstream error).
See the script's own header comment for detail.

## Related, not fixed here

While building the GEMM kernel's persistent-CTA variant, a custom `.cu` kernel reading
`gridDim.x`/`blockDim.x` directly (not through tinygrad's own codegen) got back `0`.
Root-caused via SASS disassembly (`nvcc`/`cuobjdump`, no GPU needed): `blockIdx`/`threadIdx`
are true hardware `S2R` special registers, but `blockDim`/`gridDim` are loads from a fixed
region of constant bank 0 (`c[0x0][0x0]`–`c[0x0][0x14]`) that the real CUDA driver's
`cuLaunchKernel` populates as part of its launch ABI — `NVProgram`'s direct HCQ launch path
never writes that region. Not exercised by tinygrad's own kernels (which never reference
`gridDim`/`blockDim`), so invisible to its test suite; only hits hand-written kernels using
those builtins directly. Fix location identified (`NVComputeQueue.exec()` needs to write
`local_size`/`global_size` into that constant-bank region per launch) but not implemented.
Current workaround in `nv_bf16_gemm.py`: bake the CTA count as a compile-time `#define`
instead of reading `gridDim.x` at runtime.
