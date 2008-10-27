import pycuda.driver as drv
from pytools import memoize


NVCC_OPTIONS = []




def get_arg_type(c_arg):
    if "*" in c_arg or "[" in c_arg:
        return "P"

    import re
    # remove identifier
    tp = re.sub(r"[a-zA-Z0-9]+(\[[0-9]*\])*$", "", c_arg)
    tp = tp.replace("const", "").replace("volatile", "").strip()
    if tp == "float": return "f"
    elif tp == "double": return "d"
    elif tp in ["int", "signed int"]: return "i"
    elif tp in ["unsigned", "unsigned int"]: return "I"
    elif tp in ["long", "long int"]: return "l"
    elif tp in ["unsigned long", "unsigned long int"]: return "L"
    elif tp in ["short", "short int"]: return "h"
    elif tp in ["unsigned short", "unsigned short int"]: return "H"
    elif tp in ["char"]: return "b"
    elif tp in ["unsigned char"]: return "B"
    else: raise ValueError, "unknown type '%s'" % tp
    
def get_scalar_kernel(arguments, operation, 
        name="kernel", keep=False):
    """Return a L{pycuda.driver.Function} that performs the same scalar operation
    on one or several vectors.
    """

    arguments += ", int n"
    mod = drv.SourceModule("""
        __global__ void %(name)s(%(arguments)s)
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
        options=NVCC_OPTIONS, keep=keep)

    func = mod.get_function(name)
    func.prepare(
            [get_arg_type(arg) for arg in arguments.split(",")],
            (1,1,1))

    return func


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
