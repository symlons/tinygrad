import unittest
from unittest.mock import call, patch
from typing import Any, cast

from tinygrad.runtime.ops_nv import QMD, _accessible_gpu_minors, _smem_config, nv_gpu
from tinygrad.uop.ops import KernelInfo, ProgramInfo, UOp


class _Iface:
  def __init__(self, compute_class:int): self.compute_class = compute_class


class _Device:
  def __init__(self, compute_class:int): self.iface = _Iface(compute_class)


class TestNVQMD(unittest.TestCase):
  @patch("tinygrad.runtime.ops_nv.os.close")
  @patch("tinygrad.runtime.ops_nv.os.open", side_effect=[PermissionError, 12, PermissionError])
  @patch("tinygrad.runtime.ops_nv.os.listdir", return_value=["nvidia0", "nvidia2", "nvidiactl", "nvidia7"])
  def test_accessible_gpu_minors(self, _listdir, mock_open, mock_close):
    self.assertEqual(_accessible_gpu_minors(), [2])
    self.assertEqual(mock_open.call_args_list, [call("/dev/nvidia0", 0x80002), call("/dev/nvidia2", 0x80002), call("/dev/nvidia7", 0x80002)])
    mock_close.assert_called_once_with(12)

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

  def test_dynamic_shared_memory_metadata(self):
    info = ProgramInfo.from_sink(UOp.sink(arg=KernelInfo(name="dynamic_smem", shared_mem=73728)))
    self.assertEqual(info.shared_mem, 73728)


if __name__ == "__main__": unittest.main()
