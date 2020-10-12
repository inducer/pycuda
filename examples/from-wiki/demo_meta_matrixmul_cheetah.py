#!python 
#!/usr/bin/env python
# -*- coding: utf-8 -*-


""" 
PyCuda Optimized Matrix Multiplication 
Template Meta-programming Example using Cheetah
(modified from SciPy09 Advanced Tutorial)
"""

# ------------------------------------------------------------------------------
import numpy as np
from pycuda import driver, compiler, gpuarray, tools
from Cheetah.Template import Template

import pycuda.autoinit

# -- default parameters
DEFAULT_BLOCK_SIZE = 16
DEFAULT_WORK_SIZE = 1
DEFAULT_UNROLL = 0
DEFAULT_SPILL = False
DEFAULT_PREFETCH = False

from os import path
MYPATH = path.dirname(path.abspath(__file__))
TEMPLATE_FILENAME = path.join(MYPATH, "demo_meta_matrixmul_cheetah.template.cu")

# ------------------------------------------------------------------------------
def matrixmul_opt(mat_a, mat_b, 
                  block_size = DEFAULT_BLOCK_SIZE,
                  work_size = DEFAULT_WORK_SIZE,
                  unroll = DEFAULT_UNROLL,
                  spill = DEFAULT_SPILL,
                  prefetch = DEFAULT_PREFETCH):
    
    ah, aw = mat_a.shape
    bh, bw = mat_b.shape
    
    assert aw == bh

    # -- pad input matrices appropriately
    ah_padded = int(np.ceil(ah/block_size)) * block_size
    aw_padded = int(np.ceil(aw/block_size)) * (block_size*work_size)
    mat_a_padded = np.zeros((ah_padded, aw_padded), np.float32)
    mat_a_padded[:ah,:aw] = mat_a

    bh_padded = aw_padded 
    bw_padded = int(np.ceil(bw/(block_size*work_size))) * (block_size*work_size)
    mat_b_padded = np.zeros((bh_padded, bw_padded), np.float32)
    mat_b_padded[:bh, :bw] = mat_b

    ch_padded = ah_padded
    cw_padded = bw_padded

    # -- upload padded input matrices to the GPU
    mat_a_gpu = gpuarray.to_gpu(mat_a_padded) 
    mat_b_gpu = gpuarray.to_gpu(mat_b_padded)

    # -- create empty container matrix for the result (C = A * B)
    mat_c_gpu = gpuarray.zeros((ch_padded, cw_padded), np.float32)

    # -- generate and compile the code
    # prepare the template parameters
    template_params = { 
        'BLOCK_SIZE': block_size, 
        'WORK_SIZE': work_size, 
        'UNROLL': unroll, 
        'SPILL': spill, 
        'PREFETCH': prefetch, 
        'A_WIDTH': aw_padded, 
        'A_HEIGHT': ah_padded, 
        'B_WIDTH': bw_padded,
        }
    
    # run the template engine to get the code
    kernel_code = Template(
        file = TEMPLATE_FILENAME, 
        searchList = [template_params],
        )
    
    # compile the code
    module = compiler.SourceModule(kernel_code)
    
    # get the kernel from the module
    matrixmul_func = module.get_function("matrixMul")

    # some info about the module
    print("number of registers used:", matrixmul_func.num_regs)

    # block of threads
    # ATTENTION: block is (threadDim.x, threadDim.y, threadDim.z) 
    #            and not (threadDim.z, threadDim.y, threadDim.x)
    block =  block_size, block_size, 1
    
    # grid of blocks 
    # ATTENTION: it's (blockDim.x, blockDim.y) 
    #            and not (blockDim.y, blockDim.x)
    grid = int(cw_padded/block_size/work_size), int(ch_padded/block_size)

    # -- call the kernel on the GPU
    # Note that when we use time_kernel=True pycuda will automatically synchronize the kernel 
    # to make sure that the timing is correct. If you time the code yourself, you'll have to
    # synchronize the current Context.
    gpu_time = matrixmul_func(
        # -- output
        mat_c_gpu, 
        # -- inputs
        mat_a_gpu, mat_b_gpu, 
        # -- grid of blocks
        grid = grid, 
        # -- block of threads
        block = block, 
        # -- time the kernel (approx.)
        time_kernel = True,
        )

    # get the GPU matrix back to CPU memory
    mat_c_padded = mat_c_gpu.get()
    mat_c = mat_c_padded[:ah, :bw]

    return mat_c, gpu_time

# ------------------------------------------------------------------------------
if __name__ == "__main__": 

    # matrix sizes
    a_height = 1024
    a_width = 1024
    b_height = a_width
    b_width = 1024
    
    # create random square matrices
    np.random.seed(0)
    mat_a = np.random.randn(a_height, a_width).astype(np.float32)
    mat_b = np.random.randn(b_height, b_width).astype(np.float32)
    
    # compute reference on the cpu to verify GPU computation
    mat_ref = np.dot(mat_a, mat_b)

    # -- this is a good place to auto-tune the code (using the optimization kwargs)
    # (note that you may need more that one iteration to get accurate timing estimates)
    mat_c, gpu_time = matrixmul_opt(mat_a, mat_b)

    # check for correctness
    diff = mat_c - mat_ref
    error = np.absolute(diff).max()
    assert error <= 1e-2
    l2norm = np.linalg.norm(diff)
    print("l2norm: ", l2norm)

    # print some stats
    print("gpu time:", gpu_time)
    gflop = mat_c.size * (a_width * 2.) / (1000**3.)
    gflops = gflop / gpu_time
    print("gflops:", gflops)


