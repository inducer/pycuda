#!python 
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.autoinit
import numpy as np

from pycuda.compiler import SourceModule
func_mod = SourceModule("""
template <class T>
__device__ T incr(T x) {
    return (x + 1.0);
}

// Needed to avoid name mangling so that PyCUDA can
// find the kernel function:
extern "C" {
    __global__ void func(float *a, int N)
    {
        int idx = threadIdx.x;
        if (idx < N)
            a[idx] = incr(a[idx]);
    }
}
""", no_extern_c=1)

func = func_mod.get_function('func')

N = 5
x = np.asarray(np.random.rand(N), np.float32)
x_orig = x.copy()
x_gpu = gpuarray.to_gpu(x)

func(x_gpu.gpudata, np.uint32(N), block=(N, 1, 1))
print('x:       ', x)
print('incr(x): ', x_gpu.get())

