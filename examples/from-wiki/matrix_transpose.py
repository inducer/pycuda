#!python 
# Exercise 1 from http://webapp.dam.brown.edu/wiki/SciComp/CudaExercises

# Transposition of a matrix
# by Hendrik Riedmann <riedmann@dam.brown.edu>


import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy
import numpy.linalg as la

from pycuda.tools import context_dependent_memoize

block_size = 16

@context_dependent_memoize
def _get_transpose_kernel():
    mod = SourceModule("""
    #define BLOCK_SIZE %(block_size)d
    #define A_BLOCK_STRIDE (BLOCK_SIZE * a_width)
    #define A_T_BLOCK_STRIDE (BLOCK_SIZE * a_height)

    __global__ void transpose(float *A_t, float *A, int a_width, int a_height)
    {
        // Base indices in A and A_t
        int base_idx_a   = blockIdx.x * BLOCK_SIZE +
    blockIdx.y * A_BLOCK_STRIDE;
        int base_idx_a_t = blockIdx.y * BLOCK_SIZE +
    blockIdx.x * A_T_BLOCK_STRIDE;

        // Global indices in A and A_t
        int glob_idx_a   = base_idx_a + threadIdx.x + a_width * threadIdx.y;
        int glob_idx_a_t = base_idx_a_t + threadIdx.x + a_height * threadIdx.y;

        __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE+1];

        // Store transposed submatrix to shared memory
        A_shared[threadIdx.y][threadIdx.x] = A[glob_idx_a];

        __syncthreads();

        // Write transposed submatrix to global memory
        A_t[glob_idx_a_t] = A_shared[threadIdx.x][threadIdx.y];
    }
    """% {"block_size": block_size})

    func = mod.get_function("transpose")
    func.prepare("PPii")

    from pytools import Record
    class TransposeKernelInfo(Record): pass

    return TransposeKernelInfo(func=func,
            block=(block_size, block_size, 1),
            block_size=block_size,
            granularity=block_size)



def _get_big_block_transpose_kernel():
    mod = SourceModule("""
    #define BLOCK_SIZE %(block_size)d
    #define A_BLOCK_STRIDE (BLOCK_SIZE * a_width)
    #define A_T_BLOCK_STRIDE (BLOCK_SIZE * a_height)

    __global__ void transpose(float *A, float *A_t, int a_width, int a_height)
    {
        // Base indices in A and A_t
        int base_idx_a   = 2 * blockIdx.x * BLOCK_SIZE +
    2 * blockIdx.y * A_BLOCK_STRIDE;
        int base_idx_a_t = 2 * blockIdx.y * BLOCK_SIZE +
    2 * blockIdx.x * A_T_BLOCK_STRIDE;

        // Global indices in A and A_t
        int glob_idx_a   = base_idx_a + threadIdx.x + a_width * threadIdx.y;
        int glob_idx_a_t = base_idx_a_t + threadIdx.x + a_height * threadIdx.y;

        __shared__ float A_shared[2 * BLOCK_SIZE][2 * BLOCK_SIZE + 1];

        // Store transposed submatrix to shared memory
        A_shared[threadIdx.y][threadIdx.x] = A[glob_idx_a];
        A_shared[threadIdx.y][threadIdx.x + BLOCK_SIZE] =
    A[glob_idx_a + A_BLOCK_STRIDE];
        A_shared[threadIdx.y + BLOCK_SIZE][threadIdx.x] =
    A[glob_idx_a + BLOCK_SIZE];
        A_shared[threadIdx.y + BLOCK_SIZE][threadIdx.x + BLOCK_SIZE] =
        A[glob_idx_a + BLOCK_SIZE + A_BLOCK_STRIDE];

        __syncthreads();

        // Write transposed submatrix to global memory
        A_t[glob_idx_a_t] = A_shared[threadIdx.x][threadIdx.y];
        A_t[glob_idx_a_t + A_T_BLOCK_STRIDE] =
    A_shared[threadIdx.x + BLOCK_SIZE][threadIdx.y];
        A_t[glob_idx_a_t + BLOCK_SIZE] =
    A_shared[threadIdx.x][threadIdx.y + BLOCK_SIZE];
        A_t[glob_idx_a_t + A_T_BLOCK_STRIDE + BLOCK_SIZE] =
    A_shared[threadIdx.x + BLOCK_SIZE][threadIdx.y + BLOCK_SIZE];
    }
      """% {"block_size": block_size})

    func = mod.get_function("transpose")
    func.prepare("PPii")

    from pytools import Record
    class TransposeKernelInfo(Record): pass

    return TransposeKernelInfo(func=func,
            block=(block_size, block_size, 1),
            block_size=block_size,
            granularity=2*block_size)




def _transpose(tgt, src):
    krnl = _get_transpose_kernel()

    w, h = src.shape
    assert tgt.shape == (h, w)
    assert w % krnl.granularity == 0
    assert h % krnl.granularity == 0

    krnl.func.prepared_call(
                    (w // krnl.granularity, h // krnl.granularity), krnl.block,
                    tgt.gpudata, src.gpudata, w, h)




def transpose(src):
    w, h = src.shape

    result = gpuarray.empty((h, w), dtype=src.dtype)
    _transpose(result, src)
    return result





def check_transpose():
    from pycuda.curandom import rand

    for i in numpy.arange(10, 13, 0.125):
        size = int(((2**i) // 32) * 32)
        print(size)

        source = rand((size, size), dtype=numpy.float32)

        result = transpose(source)

        err = source.get().T - result.get()
        err_norm = la.norm(err)

        source.gpudata.free()
        result.gpudata.free()

        assert err_norm == 0, (size, err_norm)




def run_benchmark():
    from pycuda.curandom import rand

    powers = numpy.arange(10, 13, 2**(-6))
    sizes = [int(size) for size in numpy.unique(2**powers // 16 * 16)]
    bandwidths = []
    times = []

    for size in sizes:

        source = rand((size, size), dtype=numpy.float32)
        target = gpuarray.empty((size, size), dtype=source.dtype)

        start = pycuda.driver.Event()
        stop = pycuda.driver.Event()

        warmup = 2

        for i in range(warmup):
            _transpose(target, source)

        count = 10

        cuda.Context.synchronize()
        start.record()

        for i in range(count):
            _transpose(target, source)

        stop.record()
        stop.synchronize()

        elapsed_seconds = stop.time_since(start)*1e-3
        mem_bw = source.nbytes / elapsed_seconds * 2 * count

        bandwidths.append(mem_bw)
        times.append(elapsed_seconds)

    slow_sizes = [s for s, bw in zip(sizes, bandwidths) if bw < 40e9]
    print(("Sizes for which bandwidth was low:", slow_sizes))
    print(("Ditto, mod 64:", [s % 64 for s in slow_sizes]))
    from matplotlib.pyplot import semilogx, loglog, show, savefig, clf, xlabel, ylabel
    xlabel('matrix size')
    ylabel('bandwidth')
    semilogx(sizes, bandwidths)
    savefig("transpose-bw.png")
    clf()
    xlabel('matrix size')
    ylabel('time')
    loglog(sizes, times)
    savefig("transpose-times.png")




#check_transpose()
run_benchmark()


