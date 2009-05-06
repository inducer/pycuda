import pycuda.driver as cuda
import pycuda.gl as cudagl
import pycuda.tools

cuda.init()
assert cuda.Device.count() >= 1

device = pycuda.tools.get_default_device()
context = cudagl.make_context(device)

import atexit
atexit.register(context.pop)
