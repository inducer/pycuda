from __future__ import absolute_import
import pycuda.driver as cuda
import pycuda.gl as cudagl
import atexit

cuda.init()
assert cuda.Device.count() >= 1

from pycuda.tools import make_default_context  # noqa: E402
context = make_default_context(lambda dev: cudagl.make_context(dev))
device = context.get_device()

atexit.register(context.pop)
