import sys

import pycuda.driver

if sys.platform == 'linux2':
    import DLFCN as dl
    flags = sys.getdlopenflags()
    sys.setdlopenflags(dl.RTLD_NOW|dl.RTLD_GLOBAL)
    from pycuda.gl import *
    sys.setdlopenflags(flags)
else:
    from pycuda.gl import *
