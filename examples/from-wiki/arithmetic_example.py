#!python 
#!python
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np

# Converting the list into numpy array for faster access and putting it into the GPU for processing... 
start = cuda.Event()
end = cuda.Event()

N = 222341

values = np.random.randn(N)
number_of_blocks=N/1024

# Calculating the (value-max)/max-min computation and storing it in a numpy array. Pre-calculating the maximum and minimum values.

# Space for the Kernel computation..

func_mod = SourceModule("""
// Needed to avoid name mangling so that PyCUDA can
// find the kernel function:
extern "C" {
__global__ void func(float *a, int N, float minval, int denom)
{
int idx = threadIdx.x+threadIdx.y*32+blockIdx.x*blockDim.x;
if (idx < N)
    a[idx] = (a[idx]-minval)/denom;
}
}
""", no_extern_c=1)

func = func_mod.get_function('func')
x = np.asarray(values, np.float32)
x_gpu = gpuarray.to_gpu(x)
h_minval = np.float32(0)
h_denom = np.int32(255)

start.record()
# a function to the GPU to calculate the computation in the GPU.
func(x_gpu.gpudata, np.uint32(N), np.float32(h_minval), np.uint32(h_denom), block=(1024, 1, 1), grid=(number_of_blocks+1,1,1))
end.record() 
end.synchronize()
secs = start.time_till(end)*1e-3

print("SourceModule time")
print("%fs" % (secs))
print('x:       ', x[N-1])
print('Func(x): ', x_gpu.get()[N-1],'Actual: ',(values[N-1]-0)/(h_denom))
x_colors=x_gpu.get()



