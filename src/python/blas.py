import pycuda.rt
from pycuda._blas import *
init()

import atexit
atexit.register(shutdown)
