#!python 
# Exercise 2 from http://webapp.dam.brown.edu/wiki/SciComp/CudaExercises

# Generate an array of random numbers between 0 and 1
# List the indices of those numbers that are greater than a given limit

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy

# Define block size and number of elements per thread
block_size = 512
el_per_thread = 1
multiple_block_size = el_per_thread * block_size

# Create an array of random numbers and set limit
amount = 256*2560
limit = 0.9
assert amount % multiple_block_size == 0
a = numpy.random.rand(amount)
a = a.astype(numpy.float32)
a_gpu = gpuarray.to_gpu(a)

# Initialize array for the selection on device
selec = numpy.zeros_like(a)
selec = selec.astype(numpy.int32)
selec.fill(-1)
selec_gpu = gpuarray.to_gpu(selec)

# Initialize a counter on device
counter_gpu = gpuarray.zeros(1, dtype=numpy.int32)

# Computation on device
mod = SourceModule("""
#define BLOCK_SIZE %(block_size)d
#define EL_PER_THREAD %(el_per_thread)d
// #define USE_LOOPS 1

__global__ void select_them(float *a, int *selec, float limit, int *counter)
{
    __shared__ int selec_smem[EL_PER_THREAD * BLOCK_SIZE];
    __shared__ int counter_smem;
    int *counter_smem_ptr;

    int jump = 16;
    int idx = EL_PER_THREAD * blockIdx.x * BLOCK_SIZE + threadIdx.x +
              (EL_PER_THREAD - 1) * (threadIdx.x / 16) * jump;

    if (threadIdx.x == 1)
    {
        counter_smem_ptr = &counter_smem;
        counter_smem = 0;
    }

    #if (EL_PER_THREAD == 1) && !defined(USE_LOOPS)
        selec_smem[threadIdx.x] = -1;
    #else
        for (int i = 0; i <= EL_PER_THREAD - 1; i++)
            selec_smem[threadIdx.x + i * BLOCK_SIZE] = -1;
    #endif

    __syncthreads();

   // each counting thread writes its index to shared memory

    #if (EL_PER_THREAD == 1) && !defined(USE_LOOPS)
        if (a[idx] >= limit)
            selec_smem[atomicAdd(counter_smem_ptr, 1)] = idx;
    #else
         for (int i = 0; i <= EL_PER_THREAD - 1; i++)
         {
             if (a[idx + i * jump] >= limit)
                 selec_smem[atomicAdd(counter_smem_ptr, 1)] = idx + i * jump;
         }
    #endif

    __syncthreads();

    if (threadIdx.x == 1)
        counter_smem = atomicAdd(counter, counter_smem);

    __syncthreads();

    #if (EL_PER_THREAD == 1) && !defined(USE_LOOPS)
        if (selec_smem[threadIdx.x] >= 0)
            selec[counter_smem + threadIdx.x] = selec_smem[threadIdx.x];
    #else
        for (int i = 0; i <= EL_PER_THREAD - 1; i++)
        {
            if (selec_smem[threadIdx.x + i * jump] >= 0)
                selec[counter_smem + threadIdx.x + i * jump] =
                selec_smem[threadIdx.x + i * jump];
        }
    #endif
}
""" % {"block_size": block_size, "el_per_thread": el_per_thread})

# Prepare function call
func = mod.get_function("select_them")
func.prepare("PPfP")

block = (block_size, 1, 1)
grid = (amount // multiple_block_size, 1)

# Warmup
warmup = 2
for i in range(warmup):
    func.prepared_call(grid, block, a_gpu.gpudata, selec_gpu.gpudata,
                       limit, counter_gpu.gpudata)
    counter_gpu = gpuarray.zeros(1, dtype=numpy.int32)

# Prepare getting the time
start = cuda.Event()
stop = cuda.Event()

# Call function and get time
cuda.Context.synchronize()
start.record()
count = 10
for i in range(count):
    func.prepared_call(grid, block, a_gpu.gpudata, selec_gpu.gpudata,
                       limit, counter_gpu.gpudata)
    counter_gpu = gpuarray.zeros(1, dtype=numpy.int32)
stop.record()

# Copy selection from device to host
selec_gpu.get(selec)

stop.synchronize()

# Evaluate memory bandwidth and verify solution
elems_in_selec = len(numpy.nonzero(selec >= 0))

elapsed_seconds = stop.time_since(start) * 1e-3
print("mem bw:", (a.nbytes + elems_in_selec * 4) / elapsed_seconds / 1e9 * count)

filtered_set = sorted(list(item for item in selec if item != -1))
reference_set = sorted(list(i for i, x in enumerate(a) if x >= limit))
assert filtered_set == reference_set

