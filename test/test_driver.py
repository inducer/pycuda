import unittest
import pycuda.driver as drv





class TestMatrices(unittest.TestCase):
    def setUp(self):
        drv.init()
        assert drv.Device.count() >= 1

        dev = drv.Device(0)
        assert isinstance(dev.name(), str)
        assert isinstance(dev.compute_capability(), tuple)
        assert isinstance(dev.get_attributes(), dict)

        self.ctx = dev.make_context()

    def test_memory(self):
        import numpy
        import numpy.linalg as la
        z = numpy.random.randn(400).astype(numpy.float32)
        new_z = drv.from_device_like(drv.to_device(z), z)
        assert la.norm(new_z-z) == 0

    def test_simple_kernel(self):
        mod = drv.SourceModule("""
        __global__ void multiply_them(float *dest, float *a, float *b)
        {
          const int i = threadIdx.x;
          dest[i] = a[i] * b[i];
        }
        """)

        multiply_them = mod.get_function("multiply_them")

        import numpy
        a = numpy.random.randn(400).astype(numpy.float32)
        b = numpy.random.randn(400).astype(numpy.float32)

        try:
            multiply_them(
                    drv.Out(numpy.zeros_like(a)), drv.In(a), drv.In(b),
                    shared=0, block=(400,1,1))
        except:
            import traceback
            traceback.print_exc()
            raise









if __name__ == "__main__":
    unittest.main()
