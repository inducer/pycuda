# Copyright 2008-2021 Andreas Kloeckner
# Copyright 2021 NVIDIA Corporation

from numba import cuda

import pycuda.driver as pycuda
# We use autoprimaryctx instead of autoinit because Numba can only operate on a
# primary context
import pycuda.autoprimaryctx  # noqa
import pycuda.gpuarray as gpuarray

import numpy


# Create a PyCUDA gpuarray
a_gpu = gpuarray.to_gpu(numpy.random.randn(4, 4).astype(numpy.float32))
print("original array:")
print(a_gpu)


# A standard Numba kernel that doubles its input array
@cuda.jit
def double(x):
    i, j = cuda.grid(2)

    if i < x.shape[0] and j < x.shape[1]:
        x[i, j] *= 2


# Call the Numba kernel on the PyCUDA gpuarray, using the CUDA Array Interface
# transparently
double[(4, 4), (1, 1)](a_gpu)
print("doubled with numba:")
print(a_gpu)
