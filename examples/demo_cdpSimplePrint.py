'''
 * demo_cdpSimplePrint.py
 *
 * Adapted from NVIDIA's "cdpSimplePrint - Simple Print (CUDA Dynamic Parallelism)" sample
 * http://docs.nvidia.com/cuda/cuda-samples/index.html#simple-print--cuda-dynamic-parallelism-
 * http://ecee.colorado.edu/~siewerts/extra/code/example_code_archive/a490dmis_code/CUDA/cuda_work/samples/0_Simple/cdpSimplePrint/cdpSimplePrint.cu
 *
 * From cdpSimplePrint.cu (not sure if this is Ok with NVIDIA's 38-page EULA though...):
 * ---------------------------------------------------------------------------
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 * ---------------------------------------------------------------------------
'''

import sys, os
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import DynamicSourceModule

cdpSimplePrint_cu = '''
#include <cstdio>

////////////////////////////////////////////////////////////////////////////////
// Variable on the GPU used to generate unique identifiers of blocks.
////////////////////////////////////////////////////////////////////////////////
__device__ int g_uids = 0;

////////////////////////////////////////////////////////////////////////////////
// Print a simple message to signal the block which is currently executing.
////////////////////////////////////////////////////////////////////////////////
__device__ void print_info( int depth, int thread, int uid, int parent_uid )
{
  if( threadIdx.x == 0 )
  {
    if( depth == 0 )
      printf( "BLOCK %d launched by the host\\n", uid );
    else
    {
      char buffer[32];
      for( int i = 0 ; i < depth ; ++i )
      {
        buffer[3*i+0] = '|';
        buffer[3*i+1] = ' ';
        buffer[3*i+2] = ' ';
      }
      buffer[3*depth] = '\\0';
      printf( "%sBLOCK %d launched by thread %d of block %d\\n", buffer, uid, thread, parent_uid );
    }
  }
  __syncthreads();
}

////////////////////////////////////////////////////////////////////////////////
// The kernel using CUDA dynamic parallelism.
//
// It generates a unique identifier for each block. Prints the information
// about that block. Finally, if the 'max_depth' has not been reached, the
// block launches new blocks directly from the GPU.
////////////////////////////////////////////////////////////////////////////////
__global__ void cdp_kernel( int max_depth, int depth, int thread, int parent_uid )
{
  // We create a unique ID per block. Thread 0 does that and shares the value with the other threads.
  __shared__ int s_uid;
  if( threadIdx.x == 0 ) 
  {
    s_uid = atomicAdd( &g_uids, 1 );
  }
  __syncthreads();

  // We print the ID of the block and information about its parent.
  print_info( depth, thread, s_uid, parent_uid );
  
  // We launch new blocks if we haven't reached the max_depth yet.
  if( ++depth >= max_depth )
  {
    return;
  }
  cdp_kernel<<<gridDim.x, blockDim.x>>>( max_depth, depth, threadIdx.x, s_uid );
}
'''

def main(argv):
    max_depth = 2
    if len(argv) > 1:
        if len(argv) == 2 and argv[1].isdigit() and int(argv[1]) >= 1 and int(argv[1]) <= 8:
            max_depth = int(argv[1])
        else:
            print("Usage: %s <max_depth>\t(where max_depth is a value between 1 and 8)." % argv[0])
            sys.exit(0)

    print("starting Simple Print (CUDA Dynamic Parallelism)")

    mod = DynamicSourceModule(cdpSimplePrint_cu)
    cdp_kernel = mod.get_function('cdp_kernel').prepare('iiii').prepared_call

    print("***************************************************************************")
    print("The CPU launches 2 blocks of 2 threads each. On the device each thread will")
    print("launch 2 blocks of 2 threads each. The GPU we will do that recursively")
    print("until it reaches max_depth=%d\n" % max_depth)
    print("In total 2")
    num_blocks, sum = 2, 2
    for i in range(1, max_depth):
        num_blocks *= 4
        print("+%d" % num_blocks)
        sum += num_blocks
    print("=%d blocks are launched!!! (%d from the GPU)" % (sum, sum-2))
    print("***************************************************************************\n")

    pycuda.autoinit.context.set_limit(cuda.limit.DEV_RUNTIME_SYNC_DEPTH, max_depth)

    print("Launching cdp_kernel() with CUDA Dynamic Parallelism:\n")
    cdp_kernel((2,1), (2,1,1), max_depth, 0, 0, -1)

if __name__ == "__main__":
    main(sys.argv)
