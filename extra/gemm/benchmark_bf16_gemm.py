import ctypes, math, os, time

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.engine.realize import get_runtime
from tinygrad.helpers import getenv
from tinygrad.runtime.ops_cuda import cu_time_execution

from extra.gemm.nv_bf16_gemm import _custom_nv_bf16_gemm, _effective_stages

M, N, K = getenv("M", 8192), getenv("N", 8192), getenv("K", 8192)
WARMUP, ITERS, SEED = getenv("WARMUP", 500), getenv("ITERS", 100), getenv("SEED", 42)
L2_BYTES, COOLDOWN = getenv("L2_MB", 40) << 20, getenv("COOLDOWN", 5)


def _ptr(t:Tensor) -> int:
  opaque = t.uop.base.buffer.ensure_allocated()._buf
  return int(opaque.value if hasattr(opaque, "value") else opaque)


def _inputs(device:str) -> tuple[list[Tensor], list[Tensor], int]:
  input_bytes = (M*K + K*N) * dtypes.bfloat16.itemsize
  groups = max(1, math.ceil(3 * L2_BYTES / input_bytes))
  Tensor.manual_seed(SEED)
  aa, bb = [], []
  for _ in range(groups):
    aa.append((Tensor.rand(M, K, dtype=dtypes.float32, device=device)*2-1).cast(dtypes.bfloat16).contiguous().realize())
    bb.append((Tensor.rand(K, N, dtype=dtypes.float32, device=device)*2-1).cast(dtypes.bfloat16).contiguous().realize())
  return aa, bb, groups


def _result(name:str, elapsed:float, groups:int, out:Tensor, extra:str=""):
  tflops = 2*M*N*K*ITERS / elapsed / 1e12
  sample = out[:1, :8].tolist()[0]
  print(f"{name:18s} {tflops:8.2f} TFLOPS  {elapsed/ITERS*1e3:7.3f} ms  groups={groups}  sample={sample}{extra}")


def benchmark_custom():
  dev = Device[Device.DEFAULT]
  aa, bb, groups = _inputs(Device.DEFAULT)
  cc = [Tensor.empty(M, N, dtype=dtypes.bfloat16, device=Device.DEFAULT).realize() for _ in range(groups)]
  program_uop = to_program(_custom_nv_bf16_gemm(cc[0].uop, aa[0].uop, bb[0].uop, Device.DEFAULT), dev.renderer)
  prg = get_runtime(Device.DEFAULT, program_uop)
  global_size, local_size = program_uop.arg.global_size, program_uop.arg.local_size
  assert global_size is not None and local_size is not None

  def run_batch(count:int, timed:bool=False) -> float:
    args = [tuple(x.uop.base.buffer.ensure_allocated()._buf for x in (cc[i%groups], aa[i%groups], bb[i%groups])) for i in range(count)]
    if not hasattr(prg, "fill_kernargs"):
      def launch_all():
        for kernargs in args: prg(*kernargs, global_size=global_size, local_size=local_size)
      elapsed = cu_time_execution(launch_all, enable=timed)
      dev.synchronize()
      return float(elapsed or 0.0)
    states = [prg.fill_kernargs(kernargs) for kernargs in args]
    q = dev.hw_compute_queue_t().wait(dev.timeline_signal, dev.timeline_value - 1).memory_barrier()
    st, en = (dev.new_signal(), dev.new_signal()) if timed else (None, None)
    if st is not None: q.timestamp(st)
    for state in states: q.exec(prg, state, global_size, local_size)
    if en is not None: q.timestamp(en)
    q.signal(dev.timeline_signal, dev.next_timeline()).submit(dev)
    dev.synchronize()
    return float(en.timestamp - st.timestamp) * 1e-6 if st is not None and en is not None else 0.0

  run_batch(WARMUP)
  elapsed = run_batch(ITERS, timed=True)
  _result(f"tinygrad NV {_effective_stages(M, N, K)}stage", elapsed, groups, cc[(ITERS-1)%groups])


class Cublas:
  def __init__(self):
    libdir = os.getenv("CUDA_LIBDIR", "/opt/cuda/13.0.2/targets/x86_64-linux/lib")
    self.lib = ctypes.CDLL(f"{libdir}/libcublas.so")
    self.handle = ctypes.c_void_p()
    self._check(self.lib.cublasCreate_v2(ctypes.byref(self.handle)))

  @staticmethod
  def _check(status:int):
    if status != 0: raise RuntimeError(f"cuBLAS status {status}")

  def gemm(self, a:Tensor, b:Tensor, c:Tensor):
    alpha, beta = ctypes.c_float(1), ctypes.c_float(0)
    # cuBLAS is column-major: B^T @ A^T writes the row-major C allocation.
    self._check(self.lib.cublasGemmEx(self.handle, 0, 0, N, M, K, ctypes.byref(alpha), ctypes.c_void_p(_ptr(b)), 14, N,
      ctypes.c_void_p(_ptr(a)), 14, K, ctypes.byref(beta), ctypes.c_void_p(_ptr(c)), 14, N, 68, -1))

  def close(self): self._check(self.lib.cublasDestroy_v2(self.handle))


def _time_cuda(run, groups:int) -> float:
  for i in range(WARMUP): run(i % groups)
  Device[Device.DEFAULT].synchronize()

  def profile():
    for i in range(ITERS): run(i % groups)

  elapsed = cu_time_execution(profile, enable=True)
  assert elapsed is not None
  return elapsed


def benchmark_vendor():
  aa, bb, groups = _inputs(Device.DEFAULT)

  cublas_out = [Tensor.empty(M, N, dtype=dtypes.bfloat16, device=Device.DEFAULT).realize() for _ in range(groups)]
  cublas = Cublas()
  elapsed = _time_cuda(lambda i: cublas.gemm(aa[i], bb[i], cublas_out[i]), groups)
  _result("cuBLAS GemmEx", elapsed, groups, cublas_out[(ITERS-1)%groups])
  cublas.close()

  if COOLDOWN: time.sleep(COOLDOWN)

  import cudnn
  handle = cudnn.create_handle()
  cudnn.set_stream(handle, 0)
  graph = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, compute_data_type=cudnn.data_type.FLOAT)
  a_desc = graph.tensor([1, M, K], [M*K, K, 1], name="A")
  b_desc = graph.tensor([1, K, N], [K*N, N, 1], name="B")
  c_desc = graph.matmul(a_desc, b_desc, name="gemm")
  c_desc.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
  graph.build([cudnn.heur_mode.A])
  workspace_size = graph.get_workspace_size()
  workspace = Tensor.empty(max(workspace_size, 1), dtype=dtypes.uint8, device=Device.DEFAULT).realize()
  cudnn_out = [Tensor.empty(M, N, dtype=dtypes.bfloat16, device=Device.DEFAULT).realize() for _ in range(groups)]

  def cudnn_gemm(i:int): graph.execute({a_desc:_ptr(aa[i]), b_desc:_ptr(bb[i]), c_desc:_ptr(cudnn_out[i])}, _ptr(workspace), handle)
  elapsed = _time_cuda(cudnn_gemm, groups)
  max_diff = (cublas_out[(ITERS-1)%groups]-cudnn_out[(ITERS-1)%groups]).abs().cast(dtypes.float32).max().item()
  _result("cuDNN graph matmul", elapsed, groups, cudnn_out[(ITERS-1)%groups],
          f"  workspace={workspace_size}  max_diff_vs_cublas={max_diff}  cuDNN={cudnn.backend_version_string()}")
  cudnn.destroy_handle(handle)


if __name__ == "__main__":
  print(f"BF16 GEMM {M}x{N}x{K}; seed={SEED}; warmup={WARMUP}; iterations={ITERS}; L2={L2_BYTES >> 20} MiB")
  if Device.DEFAULT.split(":")[0] == "NV" or getenv("CUSTOM_CUDA", 0): benchmark_custom()
  elif Device.DEFAULT.split(":")[0] == "CUDA": benchmark_vendor()
  else: raise RuntimeError("run with DEV=NV or DEV=CUDA")
