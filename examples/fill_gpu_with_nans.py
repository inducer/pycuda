from __future__ import annotations

import numpy

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray


free_bytes, total_bytes = cuda.mem_get_info()
exp = 10
while True:
    fill_floats = free_bytes // 4 - (1 << exp)
    if fill_floats < 0:
        raise RuntimeError("couldn't find allocatable size")
    print("alloc", fill_floats)
    try:
        ary = gpuarray.empty((fill_floats,), dtype=numpy.float32)
    except cuda.MemoryError:
        pass
    else:
        break

    exp += 1

ary.fill(float("nan"))

print("filled %d out of %d bytes with NaNs" % (fill_floats*4, free_bytes))
