import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda

cuda.init()
assert cuda.Device.count() >= 1

device = cuda.Device(0)
context = dev.make_context()

