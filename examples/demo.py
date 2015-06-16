# Sample source code from the Tutorial Introduction in the documentation.
from __future__ import print_function
from __future__ import absolute_import
import pycuda.driver as cuda
import pycuda.autoinit  # noqa
from pycuda.compiler import SourceModule

import numpy
a = numpy.random.randn(4, 4)

a = a.astype(numpy.float32)

a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)

cuda.memcpy_htod(a_gpu, a)

mod = SourceModule("""
    __global__ void doublify(float *a)
    {
      int idx = threadIdx.x + threadIdx.y*4;
      a[idx] *= 2;
    }
    """)

func = mod.get_function("doublify")
func(a_gpu, block=(4, 4, 1), grid=(1, 1), shared=0)

a_doubled = numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)
print("original array:")
print(a)
print("doubled with kernel:")
print(a_doubled)

# alternate kernel invocation -------------------------------------------------

func(cuda.InOut(a), block=(4, 4, 1))
print("doubled with InOut:")
print(a)

# part 2 ----------------------------------------------------------------------

import pycuda.gpuarray as gpuarray
a_gpu = gpuarray.to_gpu(numpy.random.randn(4, 4).astype(numpy.float32))
a_doubled = (2*a_gpu).get()

print("original array:")
print(a_gpu)
print("doubled with gpuarray:")
print(a_doubled)
