# Exercise 2 from http://webapp.dam.brown.edu/wiki/SciComp/CudaExercises

#generate an array of random numbers between 0 and 1#
#list the indices of those numbers, that are greater than 0.5#

from __future__ import division
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray
from pycuda.compiler import SourceModule

import numpy


#create the array of random numbers

amount = 256*2560

limit = 0.9

a = numpy.random.rand(amount)

a = a.astype(numpy.float32)

a_gpu = cuda.mem_alloc(a.nbytes)

cuda.memcpy_htod(a_gpu, a)


#initialize selec on device

selec = numpy.zeros_like(a)

#selec.fill(-1)

selec = selec.astype(numpy.int32)

selec_gpu = cuda.mem_alloc(selec.nbytes)


# Initialize a counter on device

counter = numpy.zeros(1)

counter = counter.astype(numpy.int32)

counter_gpu = cuda.mem_alloc(counter.nbytes)

cuda.memcpy_htod(counter_gpu, counter)


#Computation on device

mod = SourceModule("""
  #define BLOCK_SIZE 512

  __global__ void clean_selec(int *selec)
  {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    selec[idx] = -1;
  }


  __global__ void selec_them(float *a, int *selec, float limit, int *counter)
  {
    __shared__ int selec_smem[4 * BLOCK_SIZE];
    __shared__ int counter_smem;
    __shared__ int *counter_smem_ptr;

    int jump = 16;
    int idx = 4 * blockIdx.x * BLOCK_SIZE + threadIdx.x + 3 * (threadIdx.x/16) * jump;

    if (threadIdx.x == 1)
    {
      counter_smem_ptr = &counter_smem;
      counter_smem = 0;
    }
    #if 0
    selec_smem[threadIdx.x] = -1;
    selec_smem[threadIdx.x + BLOCK_SIZE] = -1;
    selec_smem[threadIdx.x + 2 * BLOCK_SIZE] = -1;
    selec_smem[threadIdx.x + 3 * BLOCK_SIZE] = -1;

    #else
    for (int i = 0; i <= 3; i++)
      selec_smem[threadIdx.x + i * BLOCK_SIZE] = -1;
    #endif

    __syncthreads();

   #if 1
   if (a[idx] >= limit)
     selec_smem[atomicAdd(counter_smem_ptr, 1)] = idx;    // each counting thread writes its index to shared memory

   if (a[idx + jump] >= limit)
     selec_smem[atomicAdd(counter_smem_ptr, 1)] = idx + jump;    // each counting thread writes its index to shared memory

   if (a[idx + 2 * jump] >= limit)
     selec_smem[atomicAdd(counter_smem_ptr, 1)] = idx + 2 * jump;    // each counting thread writes its index to shared memory

   if (a[idx + 3 * jump] >= limit)
     selec_smem[atomicAdd(counter_smem_ptr, 1)] = idx + 3 * jump;    // each counting thread writes its index to shared memory

    #else
    for (int i = 0; i <= 3; i++)
    {
      if (a[idx + i * jump] >= limit)
        selec_smem[atomicAdd(counter_smem_ptr, 1)] = idx + i * jump;
    }
    #endif

    __syncthreads();

    if (threadIdx.x == 1)
      counter_smem = atomicAdd(counter, counter_smem);

    __syncthreads();

    #if 1
    if (selec_smem[threadIdx.x] >= 0)
      selec[counter_smem + threadIdx.x] = selec_smem[threadIdx.x];

    if (selec_smem[threadIdx.x + jump] >= 0)
      selec[counter_smem + threadIdx.x + jump] = selec_smem[threadIdx.x + jump];

    if (selec_smem[threadIdx.x + 2 * jump] >= 0)
      selec[counter_smem + threadIdx.x + 2 * jump] = selec_smem[threadIdx.x + 2 * jump];

    if (selec_smem[threadIdx.x + 3 * jump] >= 0)
      selec[counter_smem + threadIdx.x + 3 * jump] = selec_smem[threadIdx.x + 3 * jump];

    #else
    for (int i = 0; i <= 3; i++)
    {
      if (selec_smem[threadIdx.x + i * jump] >= 0)
        selec[counter_smem + threadIdx.x + i * jump] = selec_smem[threadIdx.x + i * jump];
    }
    #endif
  }
""")

# Define block size

block_size = 512
multiple_block_size = 4 * block_size

# Clean the list before computing

clean = mod.get_function("clean_selec")
clean.prepare("P", block=(block_size,1 ,1))

clean.prepared_call((amount // block_size, 1), selec_gpu)


# Prepare function call

func = mod.get_function("selec_them")
func.prepare("PPfP", block=(block_size, 1, 1))


# Preparation for getting the time

start = pycuda.driver.Event()
stop = pycuda.driver.Event()


assert amount % multiple_block_size == 0


# Call Function and get time

cuda.Context.synchronize()
start.record()
func.prepared_call((amount // multiple_block_size, 1),
                   a_gpu, selec_gpu, limit, counter_gpu)
stop.record()


#copy selection from device to host

cuda.memcpy_dtoh(selec, selec_gpu)


# Evaluate memory bandwidth

elems_in_selec = 0
for i in range(0, amount):
    if selec[i] >= 0:
        elems_in_selec = elems_in_selec + 1

stop.synchronize()

elapsed_seconds = stop.time_since(start) * 1e-3
print "mem bw:", (a.nbytes + elems_in_selec * 4) / elapsed_seconds / 1e9

#print "a= ", a
numpy.set_printoptions(threshold=2000)

filtered_set = set(item for item in selec if item != -1)
reference_set = set(i for i, x in enumerate(a) if x>limit)
#assert filtered_set == reference_set
#print filtered_set
#print reference_set
assert filtered_set == reference_set
