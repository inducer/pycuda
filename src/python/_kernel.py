import pycuda.driver as drv
from pytools import memoize


NVCC_OPTIONS = []

def _compile_kernels(kernel):
    """compiles all kernels in this module, which is usefull for benchmarks"""
    for name in dir(kernel):
        if name.startswith("_get_") and name.endswith("_kernel"):
            if name is not "_get_scalar_kernel":
                getattr(kernel,name)()
            
                
def _get_scalar_kernel(arguments, operation, name="kernel"):
    """a function to generate c functions on the fly::
    
       basically it reduces some overhead for the programmer and
       simplifies the kernel development
    """
    mod = drv.SourceModule("""
        __global__ void %(name)s(%(arguments)s, int n)
        {

          int tid = threadIdx.x;
          int total_threads = gridDim.x*blockDim.x;
          int cta_start = blockDim.x*blockIdx.x;
          int i;
                
          for (i = cta_start + tid; i < n; i += total_threads) 
          {
            %(operation)s;
          }
        }
        """ % {
            "arguments": arguments, 
            "operation": operation,
            "name": name},
        options=NVCC_OPTIONS)

    return mod.get_function(name)

@memoize
def _get_axpbyz_kernel():
    return _get_scalar_kernel(
            "float a, float *x, float b, float *y, float *z",
            "z[i] = a*x[i] + b*y[i]",
            "axpbyz")

@memoize
def _get_axpbz_kernel():
    return _get_scalar_kernel(
            "float a, float *x,float b, float *z",
            "z[i] = a * x[i] + b",
            "axpb")

@memoize
def _get_multiply_kernel():
    return _get_scalar_kernel(
            "float *x, float *y, float *z",
            "z[i] = x[i] * y[i]",
            "multiply")

@memoize
def _get_divide_kernel():
    return _get_scalar_kernel(
            "float *x, float *y, float *z",
            "z[i] = x[i] / y[i]",
            "divide")

@memoize
def _get_rdivide_scalar_kernel():
    return _get_scalar_kernel(
            "float *x, float y, float *z",
            "z[i] = y / x[i]",
            "divide_r")



@memoize
def _get_fill_kernel():
    return _get_scalar_kernel(
            "float a, float *z",
            "z[i] = a",
            "fill")

@memoize
def _get_reverse_kernel():
    return _get_scalar_kernel(
            "float *y, float *z",
            "z[i] = y[n-1-i]",
            "reverse")

@memoize
def _get_arrange_kernel():
    return _get_scalar_kernel(
            "float *z",
            "z[i] = i",
            "arrange")


@memoize
def _get_abs_kernel():
    return _get_scalar_kernel(
            "float *y, float *z",
            "z[i] = abs(y[i])",
            "abs_method")

@memoize
def _get_pow_kernel():
    return _get_scalar_kernel(
            "float value, float *y, float *z",
            "z[i] = pow(y[i],value)",
            "pow_method")

@memoize
def _get_pow_array_kernel():
    return _get_scalar_kernel(
            "float *x, float *y, float *z",
            "z[i] = pow(x[i],y[i])",
            "pow_method")

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



__global__ void calculate_random(float *dest,int seed,int n)
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
    
    
     
    for (i = cta_start + tid; i < n; i += total_threads){
        dest[i] = mt19937sl();   
    }
              
}

    """
    )
 
    return mod.get_function("calculate_random")

@memoize
def _get_ceil_kernel():
    return _get_scalar_kernel(
            "float *y, float *z",
            "z[i] = ceilf(y[i])",
            "ceil_kernel")


@memoize
def _get_floor_kernel():
    return _get_scalar_kernel(
            "float *y, float *z",
            "z[i] = floorf(y[i])",
            "floor_kernel")
    
@memoize
def _get_fmod_kernel():
    return _get_scalar_kernel(
            "float *y, float *z, float mod",
            "z[i] = fmod(y[i],mod)",
            "fmod_kernel")
    
@memoize
def _get_ldexp_kernel():
    return _get_scalar_kernel(
            "float *y, float *z, float ix",
            "z[i] = ldexp(y[i],ix)",
            "ldexp_kernel")

@memoize
def _get_modf_kernel():
    return _get_scalar_kernel(
            "float *y, float *z,float *a",
            "z[i] = modf(y[i],&a[i])",
            "modf_kernel")

@memoize
def _get_frexp_kernel():
    return _get_scalar_kernel(
            "float *y, float *z,float *a",
            """
                int value = 0;
                z[i] = frexp(y[i],&value);
                a[i] = value;
            """,
            "frexp_kernel")

@memoize
def _get_exp_kernel():
    return _get_scalar_kernel(
            "float *y, float *z",
            "z[i] = exp(y[i])",
            "exp_kernel")    
    
@memoize
def _get_log_kernel():
    return _get_scalar_kernel(
            "float *y, float *z",
            "z[i] = log(y[i])",
            "log_kernel")    
    
@memoize
def _get_log10_kernel():
    return _get_scalar_kernel(
            "float *y, float *z",
            "z[i] = log10(y[i])",
            "log10_kernel")    
    
@memoize
def _get_sqrt_kernel():
    return _get_scalar_kernel(
            "float *y, float *z",
            "z[i] = sqrt(y[i])",
            "sqrt_kernel")    
    
    
@memoize
def _get_acos_kernel():
    return _get_scalar_kernel(
            "float *y, float *z",
            "z[i] = acos(y[i])",
            "acos_kernel")    
    
@memoize
def _get_asin_kernel():
    return _get_scalar_kernel(
            "float *y, float *z",
            "z[i] = asin(y[i])",
            "asin_kernel")    
    
@memoize
def _get_atan_kernel():
    return _get_scalar_kernel(
            "float *y, float *z",
            "z[i] = atan(y[i])",
            "atan_kernel")    
    
    
@memoize
def _get_cos_kernel():
    return _get_scalar_kernel(
            "float *y, float *z",
            "z[i] = cos(y[i])",
            "cos_kernel")    
    
@memoize
def _get_sin_kernel():
    return _get_scalar_kernel(
            "float *y, float *z",
            "z[i] = sin(y[i])",
            "sin_kernel")    
    
@memoize
def _get_tan_kernel():
    return _get_scalar_kernel(
            "float *y, float *z",
            "z[i] = tan(y[i])",
            "tan_kernel")    
        
    
@memoize
def _get_cosh_kernel():
    return _get_scalar_kernel(
            "float *y, float *z",
            "z[i] = cosh(y[i])",
            "cosh_kernel")    
    
@memoize
def _get_sinh_kernel():
    return _get_scalar_kernel(
            "float *y, float *z",
            "z[i] = sinh(y[i])",
            "sinh_kernel")    
    
@memoize
def _get_tanh_kernel():
    return _get_scalar_kernel(
            "float *y, float *z",
            "z[i] = tanh(y[i])",
            "tanh_kernel")         

@memoize
def _get_degress_kernel():
    return _get_scalar_kernel(
            "float *y, float *z, float pi",
            "z[i] = y[i] * 180 / pi",
            "degrees_kernel")         

@memoize
def _get_radians_kernel():
    return _get_scalar_kernel(
            "float *y, float *z, float pi",
            "z[i] = y[i] * pi / 180",
            "radians_kernel")         
