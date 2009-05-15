# Exercise 1 from http://webapp.dam.brown.edu/wiki/SciComp/CudaExercises

#Transposition of a Matrix A
from __future__ import division

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy


#define A on host and copy A to device

height = 4096
width = 4096

A = numpy.random.randn(height,width)

A = A.astype(numpy.float32)

A_gpu = cuda.mem_alloc(A.nbytes)

cuda.memcpy_htod(A_gpu, A)

print "Size of A in Bytes:", A.nbytes
 
#initialise A_t on host and device

A_t = numpy.zeros((width, height), dtype=A.dtype)

A_t_gpu = cuda.mem_alloc(A_t.nbytes)


#transpose A on the device

mod = SourceModule("""
  #define BLOCK_SIZE 16
  #define A_BLOCK_STRIDE (2*BLOCK_SIZE*a_width)
  #define A_T_BLOCK_STRIDE (2*BLOCK_SIZE*a_width)

  __global__ void transpose(float *A, float *A_t, int a_width, int a_height)
  {
    // Base Indices in A and A_t
    int base_idx_a   = blockIdx.x*2*BLOCK_SIZE + blockIdx.y*A_BLOCK_STRIDE;
    int base_idx_a_t = blockIdx.y*2*BLOCK_SIZE + blockIdx.x*A_T_BLOCK_STRIDE;

    // Global Indices in A and A_t
    int glob_idx_a   = base_idx_a + threadIdx.x + a_width*threadIdx.y;
    int glob_idx_a_t = base_idx_a_t + threadIdx.x + a_height*threadIdx.y;

    __shared__ float A_shared[2*BLOCK_SIZE][2*BLOCK_SIZE+1];

    // Store transposed Submatrix to shared memory
    A_shared[threadIdx.y][threadIdx.x] = A[glob_idx_a];
    A_shared[threadIdx.y][threadIdx.x+BLOCK_SIZE] = A[glob_idx_a+BLOCK_SIZE];
    A_shared[threadIdx.y+BLOCK_SIZE][threadIdx.x] = A[glob_idx_a+A_BLOCK_STRIDE];
    A_shared[threadIdx.y+BLOCK_SIZE][threadIdx.x+BLOCK_SIZE] = A[glob_idx_a+BLOCK_SIZE+A_BLOCK_STRIDE];
      
    __syncthreads();

    // Write transposed Submatrix to global memory
    A_t[glob_idx_a_t] = A_shared[threadIdx.x][threadIdx.y];
    A_t[glob_idx_a_t+A_T_BLOCK_STRIDE] = A_shared[threadIdx.x+BLOCK_SIZE][threadIdx.y];
    A_t[glob_idx_a_t+BLOCK_SIZE] = A_shared[threadIdx.x][threadIdx.y+BLOCK_SIZE];
    A_t[glob_idx_a_t+A_T_BLOCK_STRIDE+BLOCK_SIZE] = A_shared[threadIdx.x+BLOCK_SIZE][threadIdx.y+BLOCK_SIZE];
  }
  """)

# Preparation of the Function Call

block_size = 16
func = mod.get_function("transpose")
func.prepare("PPii", block=(block_size, block_size, 1))

# Preparation for getting the time

start = pycuda.driver.Event()
stop = pycuda.driver.Event()

assert width % (2*block_size) == 0
assert height % (2*block_size) == 0

# Call Function and get time

cuda.Context.synchronize()
start.record()
func.prepared_call(
		(width // (2*block_size),height // (2*block_size)), 
		A_gpu, A_t_gpu, width, height)
stop.record()

#copy A_t from device to host

cuda.memcpy_dtoh(A_t, A_t_gpu)


# Evaluate memory bandwidth

stop.synchronize()

elapsed_seconds = stop.time_since(start)*1e-3
print "mem bw:", A.nbytes / elapsed_seconds / 1e9 * 2


#print "A =", A
#print "A_t =", A_t

import numpy.linalg as la
print "errornorm =", la.norm(A.T-A_t)
