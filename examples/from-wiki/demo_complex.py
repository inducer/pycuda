#!python 
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import numpy
from pycuda.curandom import rand as curand

a = (numpy.random.randn(400)
        +1j*numpy.random.randn(400)).astype(numpy.complex64)
b = (numpy.random.randn(400)
        +1j*numpy.random.randn(400)).astype(numpy.complex64)

a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)

from pycuda.elementwise import ElementwiseKernel
complex_mul = ElementwiseKernel(
        "pycuda::complex<float> *x, pycuda::complex<float> *y, pycuda::complex<float> *z",
        "z[i] = x[i] * y[i]",
        "complex_mul",
        preamble="#include <pycuda-complex.hpp>",)

c_gpu = gpuarray.empty_like(a_gpu)
complex_mul(a_gpu, b_gpu, c_gpu)

import numpy.linalg as la
error = la.norm(c_gpu.get() - (a*b))
print(error)
assert error < 1e-5


