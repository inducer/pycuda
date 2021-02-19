# Copyright 2008-2021 Andreas Kloeckner
# Copyright 2021 NVIDIA Corporation

import pycuda.autoinit  # noqa
from pycuda.compiler import SourceModule

import cupy as cp


# Create a CuPy array (and a copy for comparison later)
cupy_a = cp.random.randn(4, 4).astype(cp.float32)
original = cupy_a.copy()


# Create a kernel
mod = SourceModule("""
    __global__ void doublify(float *a)
    {
      int idx = threadIdx.x + threadIdx.y*4;
      a[idx] *= 2;
    }
    """)

func = mod.get_function("doublify")

# Invoke PyCUDA kernel on a CuPy array
func(cupy_a, block=(4, 4, 1), grid=(1, 1), shared=0)

# Demonstrate that our CuPy array was modified in place by the PyCUDA kernel
print("original array:")
print(original)
print("doubled with kernel:")
print(cupy_a)
