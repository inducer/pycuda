import pycuda.driver as cuda
import pycuda.gl as cudagl
import pycuda.tools

cuda.init()
assert cuda.Device.count() >= 1

cudagl.init()

device = pycuda.tools.get_default_device()
context = cudagl.make_gl_context(device)

import atexit
atexit.register(context.pop)
