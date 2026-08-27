"""TileLang tuning oracle for the A100 BF16 GEMM used by nv_bf16_gemm.py.

This is deliberately not a tinygrad backend: it lets us cheaply explore tile,
warp, pipeline, and rasterization choices before porting them to the NV kernel.
"""
import argparse
import itertools

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def matmul(M:int, N:int, K:int, block_M:int, block_N:int, block_K:int,
           num_stages:int, thread_num:int, enable_rasterization:bool):
  @T.prim_func
  def main(A:T.Tensor((M, K), T.bfloat16), B:T.Tensor((N, K), T.bfloat16), C:T.Tensor((M, N), T.bfloat16)):
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
      A_shared = T.alloc_shared((block_M, block_K), T.bfloat16)
      B_shared = T.alloc_shared((block_N, block_K), T.bfloat16)
      C_local = T.alloc_fragment((block_M, block_N), T.float32)
      C_shared = T.alloc_shared((block_M, block_N), T.bfloat16)
      T.use_swizzle(panel_size=10, enable=enable_rasterization)
      T.clear(C_local)
      for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
        T.copy(A[by * block_M, k * block_K], A_shared)
        T.copy(B[bx * block_N, k * block_K], B_shared)
        T.gemm(A_shared, B_shared, C_local, transpose_B=True)
      T.copy(C_local, C_shared)
      T.copy(C_shared, C[by * block_M, bx * block_N])
  return main


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--size", type=int, default=8192)
  parser.add_argument("--quick", action="store_true")
  parser.add_argument("--source", action="store_true")
  args = parser.parse_args()
  size = args.size
  # The first entry is TileLang's own sm80 heuristic. The rest test the most
  # plausible neighboring choices without compiling the full Cartesian grid.
  configs = [
    (128, 256, 32, 2, 128, True),   # TileLang sm80 heuristic
    (128, 256, 32, 2, 256, True),
    (128, 256, 32, 3, 256, True),
    (256, 128, 32, 2, 256, True),
    (256, 128, 32, 3, 256, True),   # current NV kernel's logical tile
    (256, 128, 32, 3, 256, False),
    (128, 128, 32, 2, 128, True),
    (128, 128, 32, 3, 128, True),
    (256, 64, 32, 2, 256, True),
    (256, 64, 64, 2, 256, True),
  ]
  if not args.quick:
    configs += list(itertools.product((64, 128, 256), (64, 128, 256), (32, 64), (1, 2, 3), (128, 256), (True, False)))
  seen:set[tuple[int, int, int, int, int, bool]] = set()
  results = []
  for config in configs:
    if config in seen or size % config[0] or size % config[1]: continue
    seen.add(config)
    try:
      kernel = matmul(size, size, size, *config)
      latency = kernel.get_profiler().do_bench(warmup=25, rep=50)
      tflops = 2 * size**3 / latency * 1e-9
      print(f"{config}: {latency:.4f} ms {tflops:.2f} TFLOPS", flush=True)
      results.append((tflops, config, kernel))
    except Exception as exc:
      print(f"{config}: ERROR {type(exc).__name__}: {exc}", flush=True)
  if not results: raise RuntimeError("no TileLang configuration compiled")
  tflops, config, kernel = max(results)
  print(f"BEST {config}: {tflops:.2f} TFLOPS")
  if args.source: print(kernel.get_kernel_source())


if __name__ == "__main__": main()
