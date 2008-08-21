import pycuda.driver as drv
from pytools import memoize


NVCC_OPTIONS = []

def _compile_kernels(kernel):
    """compiles all kernels in this module, which is usefull for benchmarks"""
    for name in dir(kernel):
        if name.startswith("get_") and name.endswith("_kernel"):
            if name is not "get_scalar_kernel":
                getattr(kernel,name)()
            
                
def get_scalar_kernel(arguments, operation, name="kernel"):
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
def get_axpbyz_kernel():
    return get_scalar_kernel(
            "float a, float *x, float b, float *y, float *z",
            "z[i] = a*x[i] + b*y[i]",
            "axpbyz")

@memoize
def get_axpbz_kernel():
    return get_scalar_kernel(
            "float a, float *x,float b, float *z",
            "z[i] = a * x[i] + b",
            "axpb")

@memoize
def get_multiply_kernel():
    return get_scalar_kernel(
            "float *x, float *y, float *z",
            "z[i] = x[i] * y[i]",
            "multiply")

@memoize
def get_divide_kernel():
    return get_scalar_kernel(
            "float *x, float *y, float *z",
            "z[i] = x[i] / y[i]",
            "divide")

@memoize
def get_rdivide_scalar_kernel():
    return get_scalar_kernel(
            "float *x, float y, float *z",
            "z[i] = y / x[i]",
            "divide_r")



@memoize
def get_fill_kernel():
    return get_scalar_kernel(
            "float a, float *z",
            "z[i] = a",
            "fill")

@memoize
def get_reverse_kernel():
    return get_scalar_kernel(
            "float *y, float *z",
            "z[i] = y[n-1-i]",
            "reverse")

@memoize
def get_arange_kernel():
    return get_scalar_kernel(
            "float *z, float start, float step",
            "z[i] = start + i*step",
            "arange")


@memoize
def get_pow_kernel():
    return get_scalar_kernel(
            "float value, float *y, float *z",
            "z[i] = pow(y[i], value)",
            "pow_method")

@memoize
def get_pow_array_kernel():
    return get_scalar_kernel(
            "float *x, float *y, float *z",
            "z[i] = pow(x[i], y[i])",
            "pow_method")

@memoize
def get_fmod_kernel():
    return get_scalar_kernel(
            "float *arg, float *mod, float *z",
            "z[i] = fmod(arg[i], mod[i])",
            "fmod_kernel")
    
@memoize
def get_modf_kernel():
    return get_scalar_kernel(
            "float *x, float *intpart ,float *fracpart",
            "fracpart[i] = modf(x[i], &intpart[i])",
            "modf_kernel")

@memoize
def get_frexp_kernel():
    return get_scalar_kernel(
            "float *x, float *significand, float *exponent",
            """
                int expt = 0;
                significand[i] = frexp(x[i], &expt);
                exponent[i] = expt;
            """,
            "frexp_kernel")

@memoize
def get_ldexp_kernel():
    return get_scalar_kernel(
            "float *sig, float *expt, float *z",
            "z[i] = ldexp(sig[i], int(expt[i]))",
            "ldexp_kernel")

@memoize
def get_unary_func_kernel(func_name):
    return get_scalar_kernel(
            "float *y, float *z",
            "z[i] = %s(y[i])" % func_name,
            "%s_kernel" % func_name)    
    
@memoize
def get_dot_kernel():
    return get_scalar_kernel(
            "float *a,float *b, float *z",
            "z[i] = a[i] * b[i]",
            "dot_kernel")  
