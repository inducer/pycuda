from __future__ import annotations

import atexit

import pycuda.driver as cuda
import pycuda.gl as cudagl


cuda.init()
assert cuda.Device.count() >= 1

from pycuda.tools import make_default_context


context = make_default_context(cudagl.make_context)
device = context.get_device()

atexit.register(context.pop)
