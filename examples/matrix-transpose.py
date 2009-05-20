# Exercise 1 from http://webapp.dam.brown.edu/wiki/SciComp/CudaExercises

# Transposition of a matrix A

from __future__ import division
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.curandom as curandom
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy

block_size = 16

# Define A on host and copy A to device
height = 7488
width = 7488

A = numpy.zeros((height, width), dtype=numpy.float32)
A_gpu = curandom.rand(height*width)

# Define A_t on host and device
A_t = numpy.zeros((width, height), dtype=numpy.float32)
A_t_gpu = gpuarray.zeros_like(A_gpu)


# Transpose A on the device
mod = SourceModule("""
#define BLOCK_SIZE %(block_size)d
#define A_BLOCK_STRIDE (BLOCK_SIZE*a_width)
#define A_T_BLOCK_STRIDE (BLOCK_SIZE*a_height)

__global__ void transpose(float *A, float *A_t, int a_width, int a_height)
{
    // Base indices in A and A_t
    int base_idx_a   = 2*blockIdx.x*BLOCK_SIZE + 2*blockIdx.y*A_BLOCK_STRIDE;
    int base_idx_a_t = 2*blockIdx.y*BLOCK_SIZE + 2*blockIdx.x*A_T_BLOCK_STRIDE;

    // Global indices in A and A_t
    int glob_idx_a   = base_idx_a + threadIdx.x + a_width*threadIdx.y;
    int glob_idx_a_t = base_idx_a_t + threadIdx.x + a_height*threadIdx.y;

    __shared__ float A_shared[2*BLOCK_SIZE][2*BLOCK_SIZE+1];

    // Store transposed submatrix to shared memory
    A_shared[threadIdx.y][threadIdx.x] = A[glob_idx_a];
    A_shared[threadIdx.y][threadIdx.x+BLOCK_SIZE] = 
    A[glob_idx_a+A_BLOCK_STRIDE];
    A_shared[threadIdx.y+BLOCK_SIZE][threadIdx.x] = A[glob_idx_a+BLOCK_SIZE];
    A_shared[threadIdx.y+BLOCK_SIZE][threadIdx.x+BLOCK_SIZE] = 
    A[glob_idx_a+BLOCK_SIZE+A_BLOCK_STRIDE];
      
    __syncthreads();

    // Write transposed submatrix to global memory
    A_t[glob_idx_a_t] = A_shared[threadIdx.x][threadIdx.y];
    A_t[glob_idx_a_t+A_T_BLOCK_STRIDE] = 
    A_shared[threadIdx.x+BLOCK_SIZE][threadIdx.y];
    A_t[glob_idx_a_t+BLOCK_SIZE] = 
    A_shared[threadIdx.x][threadIdx.y+BLOCK_SIZE];
    A_t[glob_idx_a_t+A_T_BLOCK_STRIDE+BLOCK_SIZE] = 
    A_shared[threadIdx.x+BLOCK_SIZE][threadIdx.y+BLOCK_SIZE];
}
  """% {"block_size": block_size})

# Preparation of the function call
func = mod.get_function("transpose")
func.prepare("PPii", block=(block_size, block_size, 1))

# Preparation for getting the time
start = pycuda.driver.Event()
stop = pycuda.driver.Event()

assert width % (2*block_size) == 0
assert height % (2*block_size) == 0

# Warm up
func.prepared_call((width // (2*block_size),height // (2*block_size)), 
		A_gpu.gpudata, A_t_gpu.gpudata, width, height)
func.prepared_call((width // (2*block_size),height // (2*block_size)), 
		A_gpu.gpudata, A_t_gpu.gpudata, width, height)

# Call function and get time
times = 4

cuda.Context.synchronize()
start.record()
for i in range(times):
    func.prepared_call((width // (2*block_size),height // (2*block_size)), 
        		A_gpu.gpudata, A_t_gpu.gpudata, width, height)
stop.record()

# Copy A and A_t from device to host
A_t_gpu.get(A_t)
A_gpu.get(A)

# Evaluate memory bandwidth and verify solution
stop.synchronize()

elapsed_seconds = stop.time_since(start) / times * 1e-3
print "mem bw:", A_t.nbytes / elapsed_seconds / 1e9 * 2

import numpy.linalg as la
print "errornorm =", la.norm(A.T-A_t)
