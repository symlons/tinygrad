import unittest
from typing import Any, cast

from tinygrad.helpers import DEV
from tinygrad.runtime.ops_nv import QMD, _smem_config, nv_gpu


class _Iface:
  def __init__(self, compute_class:int): self.compute_class = compute_class


class _Device:
  def __init__(self, compute_class:int): self.iface = _Iface(compute_class)


class TestNVQMD(unittest.TestCase):
  def test_ampere_a_uses_qmd_2_4(self):
    qmd = QMD(cast(Any, _Device(nv_gpu.AMPERE_COMPUTE_A)), qmd_major_version=2, qmd_version=4)
    qmd.write(semaphore_release_enable0=1, release0_payload=123)
    self.assertEqual((qmd.ver, qmd.sz), (2, 0x40))
    self.assertEqual((qmd.read("qmd_major_version"), qmd.read("qmd_version")), (2, 4))
    self.assertEqual((qmd.read("semaphore_release_enable0"), qmd.read("release0_payload")), (1, 123))

  def test_ampere_b_uses_qmd_3(self):
    qmd = QMD(cast(Any, _Device(nv_gpu.AMPERE_COMPUTE_B)), qmd_major_version=3)
    qmd.write(release0_enable=1, release0_payload_lower=123)
    self.assertEqual((qmd.ver, qmd.sz), (3, 0x40))
    self.assertEqual((qmd.read("release0_enable"), qmd.read("release0_payload_lower")), (1, 123))

  def test_shared_memory_carveouts(self):
    self.assertEqual(_smem_config(2, 0x400 + 49152), (42, 42))  # GA100: target three CTAs in 164 KiB
    self.assertEqual(_smem_config(3, 0x400 + 49152), (26, 26))  # other GPUs: target two CTAs in 100 KiB
    with self.assertRaisesRegex(RuntimeError, "Too much shared memory"): _smem_config(2, 165 * 1024)


@unittest.skipUnless(DEV.interface == "MOCK" and DEV.device == "NV", "Testing NV mockgpu")
class TestNVDynamicSharedMemory(unittest.TestCase):
  # end-to-end: a kernel's declared dynamic_smem must actually reach the QMD's shared_memory_size
  # field NVProgram writes to the GPU, not just survive the KernelInfo -> TinyELF dataclass hops.
  def _build(self, dynamic_smem:int):
    from tinygrad import Device
    from tinygrad.device import TinyELF, Target
    from tinygrad.runtime.ops_nv import NVProgram
    obj = TinyELF(lib=b"\x00" * 4, name="test_dynamic_smem_kernel", target=Target(), signature=(), dynamic_smem=dynamic_smem)
    return NVProgram(Device["NV"], obj)

  def test_dynamic_smem_reaches_qmd(self):
    prog = self._build(73728)
    self.assertEqual(prog.shmem_usage, 0x400 + 73728)  # 0x400 = NVIDIA's documented 1 KiB reserved-per-block carveout
    self.assertEqual(prog.qmd.read("shared_memory_size"), prog.shmem_usage)

  def test_no_dynamic_smem_is_unaffected(self):
    prog = self._build(0)
    self.assertEqual(prog.shmem_usage, 0x400)
    self.assertEqual(prog.qmd.read("shared_memory_size"), 0x400)


if __name__ == "__main__": unittest.main()
