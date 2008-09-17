import numpy
import pycuda.driver as drv
from pytools import memoize




@memoize
def _get_random_kernel():
    """calculates random values::

       based on the algorithm found here    

       http://forums.nvidia.com/index.php?act=Attach&type=post&id=4398
    """
    
    mod = drv.SourceModule(
    """
    /*************************************************************************************
     * This is a shared memory implementation that keeps the full 625 words of state
     * in shared memory. Faster for heavy random work where you can afford the shared memory. */

    /* General shared memory version for any number of threads.
     * Note only up to 227 threads are run at any one time,
     * the rest loop and block till all are done. */
     
    #define N        624
    #define M        397
    #define INIT_MULT    1812433253    /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
    #define    ARRAY_SEED    19650218    /* Seed for initial setup before incorp array seed */
    #define MATRIX_A    0x9908b0df    /* Constant vector a */
    #define UPPER_MASK    0x80000000    /* Most significant w-r bits */
    #define LOWER_MASK    0x7fffffff    /* Least significant r bits */
    #define    TEMPER1        0x9d2c5680
    #define    TEMPER2        0xefc60000

    /* First a global memory implementation that uses 2 global reads and 1 global
     * write per result and keeps only 2 words of state in permanent shared memory. */

    __shared__ int    mtNext;        /* Start of next block of seeds */
    __shared__ int    mtNexts;    /* Indirect on above to save one global read time/call */
    __device__ int s_seeds[N];



    __device__
    int mt19937sl()
    {
        int        jj;
        int        kk;
        int    x;
        int    y;
        int        tid = threadIdx.x;
        
        kk = (mtNexts + tid) % N;
        __syncthreads();                /* Finished with mtNexts */

        if (tid == blockDim.x - 1)
        {
            mtNexts = kk + 1;            /* Will get modded on next call */
        }
        jj = 0;
        do
        {
            if (jj <= tid && tid < jj + N - M)
            {
                
                x = s_seeds[kk] & UPPER_MASK;
                if (kk < N - M)
                {
                    x |= (s_seeds[kk + 1] & LOWER_MASK);
                    y = s_seeds[kk + M];
                }
                else if (kk < N - 1)
                {
                    x |= (s_seeds[kk + 1] & LOWER_MASK);
                    y = s_seeds[kk + (M - N)];
                }
                else                /* kk == N - 1 */
                {
                    x |= (s_seeds[0] & LOWER_MASK);
                    y = s_seeds[M - 1];
                }
                y ^= x >> 1;
                if (x & 1)
                {
                    y ^= MATRIX_A;
                }
            }
            __syncthreads();            /* All done before we update */

            if (jj <= tid && tid < jj + N - M)
            {
                s_seeds[kk] = y;
            }
            __syncthreads();

        } while ((jj += N - M) < blockDim.x);

        y ^= (y >> 11);                /* Tempering */
        y ^= (y <<  7) & TEMPER1;
        y ^= (y << 15) & TEMPER2;
        y ^= (y >> 18);
        
        return y;
    }



    __global__ void calculate_random(float *dest, int seed, int n)
    {
        int tid = threadIdx.x;
        int total_threads = gridDim.x*blockDim.x;
        int cta_start = blockDim.x*blockIdx.x;
        int i;
        
        if (threadIdx.x == 0){
            mtNext = 0;
            mtNexts = seed;
            s_seeds[0] = seed;
     
            for (i = 1; i < N; i++){
                seed = (INIT_MULT * (seed ^ (seed >> 30)) + i); 
                s_seeds[i] = seed;
            }
        }
        __syncthreads();
        
        for (i = cta_start + tid; i < n; i += total_threads)
        {
            dest[i] = mt19937sl() * (1.0/4294967296.0);   
        }
    }
    """
    )
 
    func = mod.get_function("calculate_random")
    func.prepare("Pii", (1,1,1))
    return func




def rand(shape, dtype=numpy.float32):
    """Return an array of `shape` filled with random floats
    in the range [0,1).
    """
    from pycuda.gpuarray import GPUArray

    result = GPUArray(shape, dtype)

    import random
    func = _get_random_kernel()
    func.set_block_shape(*result._block)
    func.prepared_async_call(result._grid, result.stream,
            result.gpudata, random.random(), result.size)
        
    return result
