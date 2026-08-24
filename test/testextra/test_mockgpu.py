from tinygrad.helpers import DEV
import unittest, importlib

@unittest.skipUnless(DEV.interface.startswith("MOCK"), 'Testing mockgpu')
class TestMockGPU(unittest.TestCase):
  # https://github.com/tinygrad/tinygrad/pull/7627
  def test_import_typing_extensions(self):
    import test.mockgpu.mockgpu # noqa: F401  # pylint: disable=unused-import
    import typing_extensions
    importlib.reload(typing_extensions) # pytest imports typing_extension before mockgpu

  @unittest.skipUnless(DEV.interface == "MOCK" and DEV.device == "NV", "Testing NV mockgpu")
  def test_nv_device_with_remapped_minor(self):
    from tinygrad import Device
    self.assertEqual(Device["NV:5"].device, "NV:5")

if __name__ == '__main__':
  unittest.main()
