import unittest
import pycuda.driver as drv
import numpy
import numpy.linalg as la




drv.init()
assert drv.Device.count() >= 1

dev = drv.Device(0)
assert isinstance(dev.name(), str)
assert isinstance(dev.compute_capability(), tuple)
assert isinstance(dev.get_attributes(), dict)

print dev.get_attributes()

ctx = dev.make_context()




class TestMatrices(unittest.TestCase):
    def test_memory(self):
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

        dest = numpy.zeros_like(a)
        multiply_them(
                drv.Out(dest), drv.In(a), drv.In(b),
                shared=4096, block=(400,1,1))
        self.assert_(la.norm(dest-a*b) == 0)





if __name__ == "__main__":
    unittest.main()
