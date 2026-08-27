import functools, pathlib, time
import re
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.helpers import getenv
from tinygrad.renderer import Estimates
from tinygrad.uop.ops import KernelInfo, Ops, UOp

BLOCK_M_OVERRIDE, BLOCK_N, BLOCK_K = getenv("GEMM_BLOCK_M", 0), 128, getenv("GEMM_BLOCK_K", 32)
BLOCK_M = BLOCK_M_OVERRIDE or 256
if BLOCK_M not in (128, 256): raise ValueError("GEMM_BLOCK_M must be 128 or 256")
if BLOCK_K not in (32, 64): raise ValueError("GEMM_BLOCK_K must be 32 or 64")
STAGES_OVERRIDE = getenv("GEMM_STAGES", 0)
STAGES = STAGES_OVERRIDE or 4
if STAGES not in range(3, 7): raise ValueError("GEMM_STAGES must be between 3 and 6")
if BLOCK_K == 64 and (BLOCK_M != 256 or STAGES != 3): raise ValueError("GEMM_BLOCK_K=64 currently requires GEMM_BLOCK_M=256 and GEMM_STAGES=3")
SHARED_MEM = STAGES * (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N) * dtypes.bfloat16.itemsize
N_FIRST = getenv("GEMM_N_FIRST", 0)
GROUP_M = getenv("GEMM_GROUP_M", -1)
GROUP_SNAKE = getenv("GEMM_GROUP_SNAKE", 0)
CP_ASYNC_MODE = getenv("GEMM_CP_ASYNC", "cg")
SERPENTINE = getenv("GEMM_SERPENTINE", 1)
SHARED_EPILOGUE = getenv("GEMM_SHARED_EPILOGUE", 1)
ILP_128 = getenv("GEMM_ILP_128", 1)
REG_USAGE_LEVEL = getenv("GEMM_REG_USAGE_LEVEL", -1)
MAX_REGS = getenv("GEMM_MAX_REGS", 0)
SPLIT_K = getenv("GEMM_SPLIT_K", 0)
if CP_ASYNC_MODE not in ("ca", "cg", "cg128"): raise ValueError("GEMM_CP_ASYNC must be ca, cg, or cg128")
KERNEL_PATH = pathlib.Path(__file__).parent / "max_kernels" / "nv.bf16_fp32_bf16.3_stage.cu"

# Persistent-kernel dispatch: for shallow K (few k-blocks per tile), each CTA's
# work is cheap enough that the wave-transition overhead across the many CTAs
# needed to cover a big M*N dominates. A persistent kernel -- launch exactly
# SM_COUNT CTAs, each looping internally over its share of the output tiles --
# amortizes that instead of paying it per-CTA. Measured on this A100: helps by
# anywhere from +6% to +28% for K <= ~512-768 regardless of M/N/tile count
# (even at very few tiles, e.g. 8192x128x128), and *hurts* for deeper K (the
# inner K-loop no longer fully unrolls once nested in the outer persistent
# loop, up to -12% at K=8192) -- so it must stay gated on K, not on tile count.
# Only validated combined with the raw (STAGES=3, BLOCK_M=256, BLOCK_K=32)
# kernel path; PERSISTENT=1 with other configs is unvalidated territory.
SM_COUNT = getenv("GEMM_SM_COUNT", 108)  # A100-SXM4-40GB; override for other GPUs
PERSISTENT_K_THRESHOLD = getenv("GEMM_PERSISTENT_K_THRESHOLD", 512)
PERSISTENT = getenv("GEMM_PERSISTENT", -1)  # -1 = auto (gate on K), 0 = off, 1 = force on

def _stage_ptr(prefix:str, phase:str, block_m:int) -> str:
  stage_elems = block_m*BLOCK_K if prefix == "a" else BLOCK_K*BLOCK_N
  return f"(smem_{prefix}_0 + {phase} * {stage_elems})"

def _pipeline_source(src:str, block_m:int=BLOCK_M, stages:int=STAGES) -> str:
  if stages == 3 and block_m == 256: return src

  ptr_st, ptr_en = src.index("    nv_bfloat16 *smem_a_0"), src.index("\n\n    int ", src.index("    nv_bfloat16 *smem_a_0"))
  a_stage_bytes, b_stage_bytes = block_m * BLOCK_K * 2, BLOCK_K * BLOCK_N * 2
  ptrs = "\n".join([f"    nv_bfloat16 *smem_a_{i} = (nv_bfloat16 *)(smem + {i*a_stage_bytes});" for i in range(stages)] +
                   [f"    nv_bfloat16 *smem_b_{i} = (nv_bfloat16 *)(smem + {stages*a_stage_bytes+i*b_stage_bytes});" for i in range(stages)])
  src = src[:ptr_st] + ptrs + src[ptr_en:]

  prologue_st, prologue_en = src.index("    // load first tile"), src.index("    // wait on first pre-fetch load")
  loads = []
  for i in range(stages-1):
    a_loads = ([f"    __pipeline_memcpy_async(&smem_a_{i}[store_smem_a_off + (     0)], &data1[global_a_off + (    0)], 16);",
                f"    __pipeline_memcpy_async(&smem_a_{i}[store_smem_a_off + ( 64*32)], &data1[global_a_off + ( 64*K)], 16);"] if block_m == 128 else
               [f"    __pipeline_memcpy_async(&smem_a_{i}[store_smem_a_off + (     0)], &data1[global_a_off + (    0)], 16);",
                f"    __pipeline_memcpy_async(&smem_a_{i}[store_smem_a_off + ( 32*64)], &data1[global_a_off + ( 32*K)], 16);",
                f"    __pipeline_memcpy_async(&smem_a_{i}[store_smem_a_off + ( 64*64)], &data1[global_a_off + ( 64*K)], 16);",
                f"    __pipeline_memcpy_async(&smem_a_{i}[store_smem_a_off + ( 96*64)], &data1[global_a_off + ( 96*K)], 16);"])
    loads.append(f"""    // load pipeline tile {i}
{chr(10).join(a_loads)}
    __pipeline_memcpy_async(&smem_b_{i}[store_smem_b_off + (     0)], &data2[global_b_off + (    0)], 16);
    __pipeline_memcpy_async(&smem_b_{i}[store_smem_b_off + (16*128)], &data2[global_b_off + ( 16*N)], 16);
    __pipeline_commit();
    global_a_off += 32;
    global_b_off += 32 * N;

""")
  src = src[:prologue_st] + "".join(loads) + src[prologue_en:]

  phase_st, phase_en = src.index("        int phase_k = block_k % 3;"), src.index("        // load K=1 elements", src.index("        int phase_k"))
  phases = f"""        int phase_k = block_k % {stages};
        nv_bfloat16 *smem_a_curr = {_stage_ptr('a', 'phase_k', block_m)};
        nv_bfloat16 *smem_b_curr = {_stage_ptr('b', 'phase_k', block_m)};

        int next_phase_k = (block_k+1) % {stages};
        nv_bfloat16 *smem_a_next = {_stage_ptr('a', 'next_phase_k', block_m)};
        nv_bfloat16 *smem_b_next = {_stage_ptr('b', 'next_phase_k', block_m)};

        int store_phase_k = (block_k+{stages-1}) % {stages};
        nv_bfloat16 *smem_a_store = {_stage_ptr('a', 'store_phase_k', block_m)};
        nv_bfloat16 *smem_b_store = {_stage_ptr('b', 'store_phase_k', block_m)};

"""
  src = src[:phase_st] + phases + src[phase_en:]
  return src.replace("block_k < (num_k_blocks-2)", f"block_k < (num_k_blocks-{stages-1})").replace(
    "__pipeline_wait_prior(1)", f"__pipeline_wait_prior({stages-2})")

def _block_m_source(src:str, block_m:int=BLOCK_M) -> str:
  if block_m == 256: return src
  a_st, a_en = src.index("    // swizzled A\n"), src.index("    // unswizzed B\n")
  a_layout = """    // A: unswizzled 128 rows x 32 columns
    size_t global_a_off = ((grid_m * 128) * K) + ((threads % 4) * 8) + ((threads / 4) * K);
    size_t store_smem_a_off = ((threads / 4) * 32) + ((threads % 4) * 8);
    size_t load_smem_a_0_k_0 = (wg_m * 16 * 32) + ((wg_threads % 16) * 32) + ((wg_threads / 16) * 8);
    size_t load_smem_a_1_k_0 = load_smem_a_0_k_0 + (64 * 32);
    size_t load_smem_a_0_k_1 = load_smem_a_0_k_0 + 16;
    size_t load_smem_a_1_k_1 = load_smem_a_1_k_0 + 16;

"""
  src = src[:a_st] + a_layout + src[a_en:]
  removed = ("acc_frag_2_", "acc_frag_3_", "a_frag_2_", "a_frag_3_")
  lines = [line for line in src.splitlines() if not any(token in line for token in removed)]
  rewritten, seen_a_store = [], 0
  for line in lines:
    if line.lstrip().startswith("__pipeline_memcpy_async(&smem_a_store"):
      seen_a_store += 1
      if seen_a_store % 4 == 2:
        line = "            __pipeline_memcpy_async(&smem_a_store[store_smem_a_off + ( 64*32)], &data1[global_a_off + ( 64*K)], 16);"
      elif seen_a_store % 4 in (3, 0): continue
    rewritten.append(line)
  return "\n".join(rewritten).replace("((grid_m * 256) * N)", "((grid_m * 128) * N)")

def _ilp_128_source(src:str, block_m:int=BLOCK_M) -> str:
  """Use four warps for a 128x128 tile, with every warp covering all N."""
  if block_m != 128 or not ILP_128: return src
  src = src.replace("__launch_bounds__(256)", "__launch_bounds__(128)")
  src = src.replace("    int wg_n = threadIdx.z;         // 2\n"
                    "    int threads = threadIdx.x + (threadIdx.y * 32) + (threadIdx.z * 128); /* 256 */",
                    "    int wg_n = 0;                   // each warp covers all 128 columns\n"
                    "    int threads = threadIdx.x + (threadIdx.y * 32); /* 128 */")

  # The second old wg_n group is the same proven B swizzle with phase +2.
  old_b_indices = """    size_t load_smem_b_3_k_0 = load_smem_b_row + (((load_smem_b_phase + 12) ^ (threads % 8)) * 8);
    size_t load_smem_b_0_k_1 = load_smem_b_0_k_0 + (16 * 128);
    size_t load_smem_b_1_k_1 = load_smem_b_1_k_0 + (16 * 128);
    size_t load_smem_b_2_k_1 = load_smem_b_2_k_0 + (16 * 128);
    size_t load_smem_b_3_k_1 = load_smem_b_3_k_0 + (16 * 128);"""
  new_b_indices = """    size_t load_smem_b_3_k_0 = load_smem_b_row + (((load_smem_b_phase + 12) ^ (threads % 8)) * 8);
    size_t load_smem_b_4_k_0 = load_smem_b_row + (((load_smem_b_phase +  2) ^ (threads % 8)) * 8);
    size_t load_smem_b_5_k_0 = load_smem_b_row + (((load_smem_b_phase +  6) ^ (threads % 8)) * 8);
    size_t load_smem_b_6_k_0 = load_smem_b_row + (((load_smem_b_phase + 10) ^ (threads % 8)) * 8);
    size_t load_smem_b_7_k_0 = load_smem_b_row + (((load_smem_b_phase + 14) ^ (threads % 8)) * 8);
    size_t load_smem_b_0_k_1 = load_smem_b_0_k_0 + (16 * 128);
    size_t load_smem_b_1_k_1 = load_smem_b_1_k_0 + (16 * 128);
    size_t load_smem_b_2_k_1 = load_smem_b_2_k_0 + (16 * 128);
    size_t load_smem_b_3_k_1 = load_smem_b_3_k_0 + (16 * 128);
    size_t load_smem_b_4_k_1 = load_smem_b_4_k_0 + (16 * 128);
    size_t load_smem_b_5_k_1 = load_smem_b_5_k_0 + (16 * 128);
    size_t load_smem_b_6_k_1 = load_smem_b_6_k_0 + (16 * 128);
    size_t load_smem_b_7_k_1 = load_smem_b_7_k_0 + (16 * 128);"""
  assert old_b_indices in src
  src = src.replace(old_b_indices, new_b_indices)

  # Add the second half of the accumulator and B register files.
  decl_at = src.index("    // create registers for block A elements")
  accs = "\n".join(f"    float4 acc_frag_{m}_{n} = make_float4(0.0f,0.0f,0.0f,0.0f);" for m in range(2) for n in range(8, 16)) + "\n\n"
  src = src[:decl_at] + accs + src[decl_at:]
  decl_end = src.index("\n\n    __syncthreads()", src.index("    // create register for block B elements"))
  bregs = "\n" + "\n".join(f"    bf16x4 b_frag_{n}_k_{k};" for k in range(2) for n in range(8, 16))
  src = src[:decl_end] + bregs + src[decl_end:]

  # Extend each B preload from 8 to 16 fragments.
  lines, out = src.splitlines(), []
  matrix_b = re.compile(r"(\s*)__ldmatrix_b_elems\(&b_frag_6_k_([01]), &b_frag_7_k_\2, &smem_b_([a-z0-9_]+)\[load_smem_b_3_k_\2\]\);")
  for line in lines:
    out.append(line)
    if (match := matrix_b.fullmatch(line)) is not None:
      indent, k, phase = match.groups()
      for idx in range(4, 8):
        n = 8 + (idx-4)*2
        out.append(f"{indent}__ldmatrix_b_elems(&b_frag_{n}_k_{k}, &b_frag_{n+1}_k_{k}, &smem_b_{phase}[load_smem_b_{idx}_k_{k}]);")
  src = "\n".join(out)

  # Replace each original 2x8 MMA run by a 2x16 run.
  lines, out, i = src.splitlines(), [], 0
  mma = re.compile(r"(\s*)acc_frag_([01])_\d+ = __WMMA_8_16_16_bf16_float\(a_frag_\2_k_([01]), b_frag_\d+_k_\3, acc_frag_\2_\d+\);")
  while i < len(lines):
    match = mma.fullmatch(lines[i])
    if match is None:
      out.append(lines[i])
      i += 1
      continue
    indent, _, k = match.groups()
    while i < len(lines) and mma.fullmatch(lines[i]) is not None: i += 1
    for m in range(2):
      ns = range(15, -1, -1) if SERPENTINE and m & 1 else range(16)
      for n in ns:
        out.append(f"{indent}acc_frag_{m}_{n} = __WMMA_8_16_16_bf16_float(a_frag_{m}_k_{k}, b_frag_{n}_k_{k}, acc_frag_{m}_{n});")
  src = "\n".join(out)

  # A 128-thread CTA must issue four copies per lane for each operand.  The
  # old schedule issued two and left half of both shared tiles uninitialized.
  lines, out = src.splitlines(), []
  aload0 = re.compile(r"(\s*)__pipeline_memcpy_async\(&smem_a_([a-z0-9_]+)\[store_smem_a_off \+ \(\s*0\)\], "
                      r"&data1\[global_a_off \+ \(\s*0\)\], 16\);")
  aload64 = re.compile(r"\s*__pipeline_memcpy_async\(&smem_a_[a-z0-9_]+\[store_smem_a_off \+ \(\s*64\*32\)\], "
                       r"&data1\[global_a_off \+ \(\s*64\*K\)\], 16\);")
  bcopy0 = re.compile(r"(\s*)__pipeline_memcpy_async\(&smem_b_([a-z0-9_]+)\[store_smem_b_off \+ \(\s*0\)\], "
                      r"&data2\[global_b_off \+ \(\s*0\)\], 16\);")
  bcopy16 = re.compile(r"\s*__pipeline_memcpy_async\(&smem_b_[a-z0-9_]+\[store_smem_b_off \+ \(16\*128\)\], "
                       r"&data2\[global_b_off \+ \(\s*16\*N\)\], 16\);")
  for line in lines:
    if (match := aload0.fullmatch(line)) is not None:
      indent, phase = match.groups()
      for row in (0, 32, 64, 96): out.append(
        f"{indent}__pipeline_memcpy_async(&smem_a_{phase}[store_smem_a_off + ({row:3d}*32)], &data1[global_a_off + ({row:3d}*K)], 16);")
    elif aload64.fullmatch(line) is not None: continue
    elif (match := bcopy0.fullmatch(line)) is not None:
      indent, phase = match.groups()
      for row in (0, 8, 16, 24): out.append(
        f"{indent}__pipeline_memcpy_async(&smem_b_{phase}[store_smem_b_off + ({row:2d}*128)], &data2[global_b_off + ({row:2d}*N)], 16);")
    elif bcopy16.fullmatch(line) is not None: continue
    else: out.append(line)
  src = "\n".join(out)

  # Exhaustive epilogue for two M fragments and both old N groups.
  body_st, body_en = src.index("    // slower way: write accs one by one to data0"), src.rindex("\n}")
  stores = ["    // ILP-preserving 128x128 epilogue"]
  if SHARED_EPILOGUE:
    stores += ["    nv_bfloat16 *smem_c = (nv_bfloat16 *)smem;",
               "    size_t thread_c_off = ((wg_threads % 4) * 2) + (((wg_threads / 4) % 8) * 128);"]
  else:
    stores.append("    size_t thread_c_off = ((wg_threads % 4) * 2) + (((wg_threads / 4) % 8) * N);")
  ncols = (0, 8, 32, 40, 64, 72, 96, 104, 16, 24, 48, 56, 80, 88, 112, 120)
  for m in range(2):
    base = f"(wg_m * 16 + {m*64}) * 128" if SHARED_EPILOGUE else f"((grid_m * 128 + wg_m * 16 + {m*64}) * N) + (grid_n * 128)"
    stores.append(f"    size_t wg_c_off_{m} = {base};")
    for n, col in enumerate(ncols):
      for field, row, elem in (("x", 0, 0), ("y", 0, 1), ("z", 8, 0), ("w", 8, 1)):
        stride = 128 if SHARED_EPILOGUE else "N"
        dst = "smem_c" if SHARED_EPILOGUE else "data0"
        stores.append(f"    {dst}[wg_c_off_{m} + thread_c_off + ({row} * {stride}) + {col+elem}] = acc_frag_{m}_{n}.{field};")
  if SHARED_EPILOGUE:
    stores += ["", "    __syncthreads();", "    #pragma unroll", "    for (int i = 0; i < 16; i++) {",
               "        int linear = threads * 8 + i * 1024;",
               "        int row = linear / 128;", "        int col = linear % 128;",
               "        *reinterpret_cast<uint4 *>(&data0[(grid_m * 128 + row) * N + grid_n * 128 + col]) =",
               "          *reinterpret_cast<uint4 *>(&smem_c[linear]);", "    }"]
  return src[:body_st] + "\n".join(stores) + src[body_en:]

def _serpentine_mma_source(src:str) -> str:
  # cuBLAS traverses adjacent N fragments in opposite directions for successive
  # M fragments. This keeps the last B operand live for the next MMA and avoids
  # a needless operand-collector turnover at every row boundary.
  lines, out, i = src.splitlines(), [], 0
  while i < len(lines):
    if re.match(r"\s*acc_frag_\d+_\d+ = __WMMA", lines[i]) is None:
      out.append(lines[i])
      i += 1
      continue
    run = []
    while i < len(lines) and re.match(r"\s*acc_frag_\d+_\d+ = __WMMA", lines[i]):
      run.append(lines[i])
      i += 1
    grouped:dict[int, list[str]] = {}
    for line in run: grouped.setdefault(int(re.search(r"acc_frag_(\d+)_", line).group(1)), []).append(line)  # type: ignore[union-attr]
    for m, group in grouped.items(): out.extend(reversed(group) if m & 1 else group)
  return "\n".join(out)

def _fragment_loads(suffix:int, phase:str, half:int=0) -> str:
  aoff, boff = half * 256 * 32, half * 32 * 128
  aadd, badd = (f" + {aoff}" if aoff else ""), (f" + {boff}" if boff else "")
  return "\n".join(
    [f"        __ldmatrix_a_elems(&a_frag_{i}_k_{suffix},                &smem_a_{phase}[load_smem_a_{i}_k_{suffix}{aadd}]);" for i in range(4)] +
    [f"        __ldmatrix_b_elems(&b_frag_{i}_k_{suffix}, &b_frag_{i+1}_k_{suffix}, &smem_b_{phase}[load_smem_b_{i//2}_k_{suffix}{badd}]);"
     for i in range(0, 8, 2)])

def _async_stage_load(stage:str, indent:str="    ", conditional:bool=False) -> str:
  pred = "            " if conditional else indent
  lines = []
  for half in range(2):
    ash, bsh, ag = half * 256 * 32, half * 32 * 128, half * 32
    for row in (0, 32, 64, 96):
      lines.append(f"{pred}__pipeline_memcpy_async(&smem_a_{stage}[store_smem_a_off + ({ash + row*64:6d})], "
                   f"&data1[global_a_off + ({ag:2d}) + ({row:3d}*K)], 16);")
    for row in (0, 16):
      lines.append(f"{pred}__pipeline_memcpy_async(&smem_b_{stage}[store_smem_b_off + ({bsh + row*128:5d})], "
                   f"&data2[global_b_off + ({half*32 + row:2d}*N)], 16);")
  return "\n".join(lines)

def _block_k_source(src:str) -> str:
  if BLOCK_K == 32: return src
  # A K=64 stage is two adjacent copies of the existing swizzled K=32 layout.
  ptr_st = src.index("    nv_bfloat16 *smem_a_0")
  ptr_en = src.index("\n\n", src.index("    nv_bfloat16 *smem_b_2", ptr_st))
  ptrs = "\n".join([
    "    nv_bfloat16 *smem_a_0 = (nv_bfloat16 *)(smem);",
    "    nv_bfloat16 *smem_a_1 = (nv_bfloat16 *)(smem + 32768);",
    "    nv_bfloat16 *smem_a_2 = (nv_bfloat16 *)(smem + 65536);",
    "    nv_bfloat16 *smem_b_0 = (nv_bfloat16 *)(smem + 98304);",
    "    nv_bfloat16 *smem_b_1 = (nv_bfloat16 *)(smem + 114688);",
    "    nv_bfloat16 *smem_b_2 = (nv_bfloat16 *)(smem + 131072);",
  ])
  src = src[:ptr_st] + ptrs + src[ptr_en:]
  src = src.replace("int num_k_blocks = K / 32;", "int num_k_blocks = K / 64;")

  prologue_st, prologue_en = src.index("    // load first tile"), src.index("    // wait on first pre-fetch load")
  prologue = f"""    // preload two K=64 stages
{_async_stage_load('0')}
    __pipeline_commit();
    global_a_off += 64;
    global_b_off += 64 * N;
{_async_stage_load('1')}
    __pipeline_commit();
    global_a_off += 64;
    global_b_off += 64 * N;

"""
  src = src[:prologue_st] + prologue + src[prologue_en:]

  body_st = src.index("        // load K=1 elements for the current tile")
  mma0_st = src.index("        // MMA K=0", body_st)
  copy_st = src.index("        // load next tile", mma0_st)
  mma1_st = src.index("        // MMA K=1", copy_st)
  body_en = src.index("\n    }\n\n    // write accumulators", mma1_st)
  mma0, mma1 = src[mma0_st:copy_st], src[mma1_st:body_en]
  copy = f"""        // enqueue the next K=64 stage while the final register fragment computes
        if (block_k < (num_k_blocks-2)) {{
{_async_stage_load('store', conditional=True)}
            global_a_off += 64;
            global_b_off += 64 * N;
        }}
        __pipeline_commit();
        __pipeline_wait_prior(1);
        __syncthreads();

        // preload K=0 of the next stage before consuming the final current fragment
{_fragment_loads(0, 'next')}

"""
  body = f"""        // K=16..31
{_fragment_loads(1, 'curr')}
{mma0.replace('// MMA K=0', '// MMA K=0..15')}

        // K=32..47; reuse the K=0 register bank after its MMA consumers
{_fragment_loads(0, 'curr', 1)}
{mma1.replace('// MMA K=1', '// MMA K=16..31')}

        // K=48..63; reuse the K=1 register bank
{_fragment_loads(1, 'curr', 1)}
{mma0.replace('// MMA K=0', '// MMA K=32..47')}

{copy}{mma1.replace('// MMA K=1', '// MMA K=48..63')}"""
  return src[:body_st] + body + src[body_en:]

def _shared_epilogue_source(src:str, block_m:int=BLOCK_M) -> str:
  if not SHARED_EPILOGUE or block_m != 256: return src
  body_st = src.index("    // slower way: write accs one by one to data0")
  body_en = src.rindex("\n}")
  body = src[body_st:body_en]
  body = body.replace("size_t wg_c_off     = ((grid_m * 256) * N) + (grid_n * 128) + (wg_m * 16 * N) + (wg_n * 16);",
                      "nv_bfloat16 *smem_c = (nv_bfloat16 *)smem;\n    size_t wg_c_off = (wg_m * 16 * 128) + (wg_n * 16);")
  body = body.replace("size_t thread_c_off = ((wg_threads % 4) * 2) + (((wg_threads / 4) % 8) * N);",
                      "size_t thread_c_off = ((wg_threads % 4) * 2) + (((wg_threads / 4) % 8) * 128);")
  body = body.replace("data0[", "smem_c[").replace("(8 * N)", "(8 * 128)").replace("64*N", "64*128")
  vector_store = """
    __syncthreads();
    // Coalesced 128-bit epilogue: each lane writes 16 naturally aligned vectors.
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int linear = threads * 8 + i * 2048;
        int row = linear / 128;
        int col = linear % 128;
        *reinterpret_cast<uint4 *>(&data0[(grid_m * 256 + row) * N + grid_n * 128 + col]) =
          *reinterpret_cast<uint4 *>(&smem_c[linear]);
    }
"""
  return src[:body_st] + body + vector_store + src[body_en:]

def _persistent_source(src:str, num_ctas:int, num_k_blocks_expr:str="K / 32") -> str:
  # Wrap the whole per-tile body (grid_m/grid_n derivation through the epilogue's
  # final store) in a grid-stride loop over a fixed NUM_CTAS launch, so each CTA
  # handles multiple output tiles without paying a fresh CTA-launch/prologue cost
  # per tile. NOTE: tinygrad's NV backend does not populate `gridDim` inside
  # kernels (reads as 0, confirmed with a probe kernel) -- use the compile-time
  # NUM_CTAS constant instead of `gridDim.x`, or this hangs the GPU.
  # num_k_blocks_expr lets a split-K caller override the loop depth -- this function
  # wholesale-replaces the header region (including the original num_k_blocks line),
  # so _split_k_source's edit to that line would otherwise be silently discarded.
  head_st = src.index('    int grid_m = blockIdx.x;        /* M//256 */')
  head_en = src.index('    // ldmatrix indices')
  body_en = src.index('\n}\n', src.rindex('data0['))
  header_repl = f"""    int wg_threads = threadIdx.x;   // 32
    int wg_m = threadIdx.y;         // 4
    int wg_n = threadIdx.z;         // 2
    int threads = threadIdx.x + (threadIdx.y * 32) + (threadIdx.z * 128); /* 256 */
    int num_k_blocks = {num_k_blocks_expr};
    int num_pid_m = M / 256, num_pid_n = N / 128;
    int total_tiles = num_pid_m * num_pid_n;

    for (int pid = blockIdx.x; pid < total_tiles; pid += {num_ctas}) {{
    int grid_m = pid / num_pid_n;
    int grid_n = pid % num_pid_n;

"""
  body = src[head_en:body_en]
  body_indented = "\n".join(("    " + l if l.strip() else l) for l in body.split("\n"))
  new_src = src[:head_st] + header_repl + body_indented + "\n    __syncthreads();\n    }\n" + src[body_en+3:]
  return new_src.rstrip("\n") + "\n}\n"

def _split_k_source(src:str, k_start:int, num_k_blocks:int) -> str:
  # Address just this piece's K-slice: override the loop trip count (normally
  # derived from the full K) with a fixed chunk depth, and add a column/row
  # offset to A/B's base addresses. The row-stride math (which still
  # multiplies by the full K/N macros) is untouched -- each row of A truly
  # has stride K in memory regardless of which columns we read, so no
  # pointer-level tricks or buffer copies are needed, just these two
  # compile-time constants. Composes with GROUP_M/N_FIRST/STAGES/etc since
  # none of those touch num_k_blocks or the two lines patched here.
  marker_nb = "    int num_k_blocks = K / 32;"
  marker_a = ("    size_t global_a_off = ((grid_m * 256) * K) + ((threads %  4) * 8) + "
              "(((threads /  4) % 2) * 8 * 16 * K) + ((threads / 8) * K);")
  marker_b = "    size_t global_b_off = (grid_n * 128) + ((threads % 16) * 8) + ((threads / 16) * N);"
  for m in (marker_nb, marker_a, marker_b): assert m in src, f"split-K marker not found: {m!r}"
  src = src.replace(marker_nb, f"    int num_k_blocks = {num_k_blocks};")
  src = src.replace(marker_a, marker_a[:-1] + f" + {k_start};")
  src = src.replace(marker_b, marker_b[:-1] + f" + ({k_start} * N);")
  return src

def _effective_block_m(m:int, n:int, k_depth:int) -> int:
  # Keep explicitly forced persistent/split-K and K=64 configurations on the
  # 256-row tile their source transforms require.
  auto_128 = not BLOCK_M_OVERRIDE and BLOCK_K == 32 and SPLIT_K < 2 and PERSISTENT != 1
  if not auto_128 or k_depth <= PERSISTENT_K_THRESHOLD: return BLOCK_M
  if n == BLOCK_N: return 128  # single N-tile (thin-N): proven four-warp ILP win regardless of M
  # Otherwise only worth it if halving the M-tile keeps the whole grid within one wave --
  # crossing just above SM_COUNT costs more in wave-quantization overhead than it gains.
  # Confirmed on hardware: thin-M=256 (64->128 CTAs, crosses just above 108) regresses 32%,
  # while 1024^3 (32->64 CTAs, stays under 108) gains 20-33%.
  return 128 if (m // 128) * (n // BLOCK_N) <= SM_COUNT else BLOCK_M

def _effective_stages(m:int, n:int, k_depth:int) -> int:
  return 6 if _effective_block_m(m, n, k_depth) == 128 and ILP_128 and not STAGES_OVERRIDE else STAGES

@functools.cache
def _custom_nv_bf16_gemm(C:UOp, A:UOp, B:UOp, dname:str, k_start:int=0, k_chunk:int=0) -> UOp:
  M, K = A.shape
  K2, N = B.shape
  assert K == K2 and C.shape == (M, N)
  is_split_k = k_chunk > 0
  k_eff = k_chunk if is_split_k else K  # this invocation's own K depth, for cost/dispatch decisions
  # A deep-K single-N-tile GEMM needs more output CTAs.  Use the 128x128
  # four-warp ILP path automatically unless GEMM_BLOCK_M explicitly pins a
  # tile size.  Shallow K keeps the proven 256x128 persistent path.
  block_m = _effective_block_m(M, N, k_eff)
  ilp_128 = block_m == 128 and bool(ILP_128)
  stages = _effective_stages(M, N, k_eff)
  shared_mem = stages * (block_m*BLOCK_K + BLOCK_K*BLOCK_N) * dtypes.bfloat16.itemsize
  # persistent dispatch only validated on the raw (STAGES=3, BLOCK_M=256, BLOCK_K=32) path.
  # Gate on this invocation's own depth (k_eff) so a split-K piece can independently
  # qualify once its chunk is shallow enough, even if the pre-split K wasn't.
  use_persistent = bool(PERSISTENT == 1 or (PERSISTENT < 0 and k_eff <= PERSISTENT_K_THRESHOLD and block_m == 256 and BLOCK_K == 32))
  group_m = 0 if use_persistent else (GROUP_M if GROUP_M >= 0 else (2 if M < 4096 else 8))
  grid_tag = "persist" if use_persistent else (f"g{group_m}" if group_m else ('nfirst' if N_FIRST else 'mfirst'))
  if ilp_128: grid_tag += "_ilp"
  split_tag = f"_sk{k_start}_{k_chunk}" if is_split_k else ""
  name = f"nv_bf16_gemm_{M}_{N}_{K}_bk{BLOCK_K}_s{stages}_{grid_tag}{split_tag}"
  tx, ty = UOp.special(32, "lidx0"), UOp.special(4, "lidx1")
  tz = UOp.special(1 if ilp_128 else 2, "lidx2")
  if use_persistent: gx, gy = UOp.special(SM_COUNT, "gidx0"), UOp.special(1, "gidx1")
  elif group_m: gx, gy = UOp.special((M//block_m)*(N//BLOCK_N), "gidx0"), UOp.special(1, "gidx1")
  elif N_FIRST: gx, gy = UOp.special(N//BLOCK_N, "gidx0"), UOp.special(M//block_m, "gidx1")
  else: gx, gy = UOp.special(M//block_m, "gidx0"), UOp.special(N//BLOCK_N, "gidx1")
  sink = UOp.sink(C.base, A.base, B.base, tx, ty, tz, gx, gy,
    arg=KernelInfo(name, estimates=Estimates(ops=2*M*N*k_eff, mem=(M*k_eff+k_eff*N+M*N)*2), dynamic_smem=shared_mem))
  src = KERNEL_PATH.read_text().replace(", int N, int K)", ")").replace("wmma_example", name)
  src = src.replace("#define N_PAD 132", f"#define M {M}\n#define N {N}\n#define K {K}\n#define N_PAD 132")
  if use_persistent:
    pass  # keep the raw grid_m/grid_n lines; _persistent_source rewrites them below
  elif group_m:
    grid = f"""int pid = blockIdx.x;
    int num_pid_m = M / {block_m}, num_pid_n = N / {BLOCK_N};
    int num_pid_in_group = {group_m} * num_pid_n;
    int group_id = pid / num_pid_in_group;
    int first_pid_m = group_id * {group_m};
    int group_size_m = min(num_pid_m - first_pid_m, {group_m});
    int grid_m = first_pid_m + (pid % group_size_m);
    int grid_n = (pid % num_pid_in_group) / group_size_m;
    {"if (group_id & 1) grid_n = num_pid_n - 1 - grid_n;" if GROUP_SNAKE else ""}"""
    src = src.replace("int grid_m = blockIdx.x;        /* M//256 */\n    int grid_n = blockIdx.y;        /* N//128 */", grid)
  elif N_FIRST:
    src = src.replace("int grid_m = blockIdx.x;", "int grid_m = blockIdx.y;").replace("int grid_n = blockIdx.y;", "int grid_n = blockIdx.x;")
  if SERPENTINE and not use_persistent: src = _serpentine_mma_source(src)  # serpentine x persistent is untested; skip
  if use_persistent:
    src = _shared_epilogue_source(src)
    # split-K's marker strings are matched pre-indentation; must apply before _persistent_source
    # re-indents the whole body (adding a 4-space prefix to every line) or the markers won't match.
    # _persistent_source's num_k_blocks patch (below) is what actually takes effect for that one
    # line -- it wholesale-replaces the header region _split_k_source's edit to it would land in.
    if is_split_k: src = _split_k_source(src, k_start, k_chunk // BLOCK_K)
    src = _persistent_source(src, SM_COUNT, num_k_blocks_expr=str(k_chunk // BLOCK_K) if is_split_k else "K / 32")
  else:
    src = _ilp_128_source(_shared_epilogue_source(
      _block_k_source(_pipeline_source(_block_m_source(src, block_m), block_m, stages)), block_m), block_m)
    if is_split_k: src = _split_k_source(src, k_start, k_chunk // BLOCK_K)
  if CP_ASYNC_MODE != "cg":
    # base file's __pipeline_memcpy_async hardcodes cp.async.cg; patch the mnemonic for ca/cg128.
    op = "cp.async.ca.shared.global" if CP_ASYNC_MODE == "ca" else "cp.async.cg.shared.global.L2::128B"
    marker = 'asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n"'
    assert marker in src, "expected __pipeline_memcpy_async's cp.async.cg asm not found"
    src = src.replace(marker, f'asm volatile("{op} [%0], [%1], 16;\\n"')
  compiler = Device[dname].compiler
  auto_max_regs = MAX_REGS or ({4096:228, 8192:229}.get(M, 0) if M == N == K else 0)
  if REG_USAGE_LEVEL >= 0 or auto_max_regs:
    from tinygrad.runtime.support.compiler_cuda import NVCCCompiler
    extra_options = ([f"-Xptxas=-regUsageLevel={REG_USAGE_LEVEL}"] if REG_USAGE_LEVEL >= 0 else []) + \
                    ([f"-maxrregcount={auto_max_regs}"] if auto_max_regs else [])
    compiler = NVCCCompiler(compiler.arch, ptx=compiler.ptx, cache_key="nv_gemm",  # type: ignore[attr-defined]
                            extra_options=extra_options)
  binary = compiler.compile_cached(src)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)))

# Split-K: for deep-K shapes whose (M//BLOCK_M)*(N//BLOCK_N) tile count is well
# below the SM count, no amount of K-depth tuning raises occupancy -- there
# just aren't enough independent output tiles to fill the GPU. Splitting K
# across SPLIT_K independent kernel launches (each a normal, unmodified-tile
# invocation over a K-slice, combined via an FP32 sum) multiplies the tile
# count by SPLIT_K without touching the proven tile/warp/swizzle code at all:
# each slice's K-loop bound and A/B base address get a compile-time override
# (see _split_k_source) instead of a buffer copy, so there's no extra memory
# traffic beyond the actual FLOP-equivalent work. Off by default (0); set
# GEMM_SPLIT_K to an explicit factor -- no auto-heuristic yet, unlike
# GEMM_PERSISTENT_K_THRESHOLD, since the right factor depends on tile count
# in a way that hasn't been tuned into a general rule yet.
def nv_bf16_gemm(a:Tensor, b:Tensor) -> Tensor:
  """A100-tuned BF16 GEMM with FP32 accumulation and BF16 output."""
  if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]: raise ValueError(f"invalid GEMM shapes {a.shape} @ {b.shape}")
  if a.dtype != dtypes.bfloat16 or b.dtype != dtypes.bfloat16: raise ValueError("nv_bf16_gemm requires BF16 inputs")
  if a.device != b.device or not isinstance(a.device, str) or a.device.split(":")[0] != "NV":
    raise ValueError("nv_bf16_gemm requires both inputs on the same NV device")
  M, K, N = a.shape[0], a.shape[1], b.shape[1]
  block_m = _effective_block_m(M, N, K)
  min_k = max(3, _effective_stages(M, N, K)-1) * BLOCK_K
  if M%block_m or N%BLOCK_N or K%BLOCK_K or K < min_k:
    raise ValueError(f"shapes must be divisible by ({block_m}, {BLOCK_N}, {BLOCK_K}) and K must be at least {min_k}")
  if SPLIT_K >= 2:
    k_chunk = K // SPLIT_K
    min_chunk = max(3, STAGES-1) * BLOCK_K
    if K % SPLIT_K or k_chunk % BLOCK_K or k_chunk < min_chunk:
      raise ValueError(f"K={K} isn't evenly splittable into {SPLIT_K} pieces that are multiples of {BLOCK_K} and >= {min_chunk}")
    partials = []
    for i in range(SPLIT_K):
      out_i = Tensor.empty(M, N, dtype=dtypes.bfloat16, device=a.device)
      fxn = functools.partial(_custom_nv_bf16_gemm, dname=a.device, k_start=i*k_chunk, k_chunk=k_chunk)
      partials.append(Tensor.custom_kernel(out_i, a, b, fxn=fxn)[0])
    return Tensor.stack(*partials).cast(dtypes.float32).sum(axis=0).cast(dtypes.bfloat16)
  out = Tensor.empty(M, N, dtype=dtypes.bfloat16, device=a.device)
  return Tensor.custom_kernel(out, a, b, fxn=functools.partial(_custom_nv_bf16_gemm, dname=a.device))[0]

if __name__ == "__main__":
  M, N, K, cnt = getenv("M", 8192), getenv("N", 8192), getenv("K", 8192), getenv("CNT", 20)
  a = Tensor.ones(M, K, dtype=dtypes.bfloat16).realize()
  b = Tensor.ones(K, N, dtype=dtypes.bfloat16).realize()
  mm = TinyJit(lambda x,y: nv_bf16_gemm(x, y).realize())
  for _ in range(3): out = mm(a, b)
  Device[Device.DEFAULT].synchronize()
  st = time.perf_counter()
  for _ in range(cnt): out = mm(a, b)
  Device[Device.DEFAULT].synchronize()
  elapsed = time.perf_counter() - st
  print(f"{2*M*N*K*cnt/elapsed/1e12:.2f} TFLOPS, {elapsed/cnt*1e3:.3f} ms, mismatches={(out != K).sum().item()}")
