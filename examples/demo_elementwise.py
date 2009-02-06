import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy

a_gpu = gpuarray.to_gpu(numpy.random.randn(50).astype(numpy.float32))
b_gpu = gpuarray.to_gpu(numpy.random.randn(50).astype(numpy.float32))

from pycuda.elementwise import get_scalar_kernel
lin_comb = get_scalar_kernel(
        "float a, float *x, float b, float *y, float *z",
        "z[i] = a*x[i] + b*y[i]",
        "linear_combination")

c_gpu = gpuarray.empty_like(a_gpu)
lin_comb.set_block_shape(*a_gpu._block)
lin_comb.prepared_call(a_gpu._grid, 5, a_gpu.gpudata, 6, b_gpu.gpudata, c_gpu.gpudata, a_gpu.mem_size)

import numpy.linalg as la
assert la.norm((c_gpu - (5*a_gpu+6*b_gpu)).get()) < 1e-5
