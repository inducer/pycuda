import pycuda.driver as cuda

# Initialize CUDA
cuda.init()

from pycuda.tools import make_default_context
context = make_default_context()
device = context.get_device()

import atexit
atexit.register(context.pop)
