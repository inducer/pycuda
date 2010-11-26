import pycuda.driver as cuda
import pycuda.gl as cudagl
import pycuda.tools

cuda.init()
assert cuda.Device.count() >= 1

# TODO: get default device
device = cuda.Device(0)
context = cudagl.make_context(device)

import atexit
atexit.register(context.pop)
