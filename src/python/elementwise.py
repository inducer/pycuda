import pycuda.driver as drv
from pytools import memoize
import numpy




@memoize
def is_64_bit_platform():
    from struct import calcsize
    return calcsize('l') == 8




def dtype_to_ctype(dtype):
    dtype = numpy.dtype(dtype)
    if dtype == numpy.int64 and is_64_bit_platform():
        return "long"
    elif dtype == numpy.uint64 and is_64_bit_platform():
        return "unsinged long"
    elif dtype == numpy.int32:
        return "int"
    elif dtype == numpy.uint32:
        return "unsigned int"
    elif dtype == numpy.int16:
        return "short int"
    elif dtype == numpy.uint16:
        return "short unsigned int"
    elif dtype == numpy.int8:
        return "signed char"
    elif dtype == numpy.uint8:
        return "unsigned char"
    elif dtype == numpy.float32:
        return "float"
    elif dtype == numpy.float64:
        return "double"
    else:
        raise ValueError, "unable to map dtype '%s'" % dtype




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
    
def get_elwise_module(arguments, operation, 
        name="kernel", keep=False, options=[], preamble=""):
    from pycuda.compiler import SourceModule
    return SourceModule("""
        %(preamble)s

        __global__ void %(name)s(%(arguments)s)
        {

          unsigned tid = threadIdx.x;
          unsigned total_threads = gridDim.x*blockDim.x;
          unsigned cta_start = blockDim.x*blockIdx.x;
          unsigned i;
                
          for (i = cta_start + tid; i < n; i += total_threads) 
          {
            %(operation)s;
          }
        }
        """ % {
            "arguments": arguments, 
            "operation": operation,
            "name": name,
            "preamble": preamble},
        options=options, keep=keep)

def get_elwise_kernel_and_types(arguments, operation, 
        name="kernel", keep=False, options=[]):
    arguments += ", unsigned n"
    mod = get_elwise_module(arguments, operation, name,
            keep, options)

    func = mod.get_function(name)
    arg_types = [get_arg_type(arg) for arg in arguments.split(",")]
    func.prepare("".join(arg_types), (1,1,1))

    return func, arg_types

def get_elwise_kernel(arguments, operation, 
        name="kernel", keep=False, options=[]):
    """Return a L{pycuda.driver.Function} that performs the same scalar operation
    on one or several vectors.
    """
    func, arg_types = get_elwise_kernel_and_types(
            arguments, operation, name, keep, options)

    return func




class ElementwiseKernel:
    def __init__(self, arguments, operation, 
            name="kernel", keep=False, options=[]):
        self.func, self.arg_types = get_elwise_kernel_and_types(
            arguments, operation, name, keep, options)

        assert [i for i, arg_tp in enumerate(self.arg_types) if arg_tp == "P"], \
                "ElementwiseKernel can only be used with functions that have at least one " \
                "vector argument"

    def __call__(self, *args):
        from pytools import single_valued
        vectors = []

        invocation_args = []
        for arg, arg_tp in zip(args, self.arg_types):
            if arg_tp == "P":
                vectors.append(arg)
                invocation_args.append(arg.gpudata)
            else:
                invocation_args.append(arg)

        repr_vec = vectors[0]
        invocation_args.append(repr_vec.mem_size)
        self.func.set_block_shape(*repr_vec._block)
        self.func.prepared_call(repr_vec._grid, *invocation_args)




@memoize
def get_take_kernel(dtype, idx_dtype):
    ctx = {
            "idx_tp": dtype_to_ctype(idx_dtype),
            "tp": dtype_to_ctype(dtype),
            }

    mod = get_elwise_module(
            "%(idx_tp)s *idx, %(tp)s *dest, unsigned n" % ctx,
            "dest[i] = tex1Dfetch(tex_src, idx[i])",
            "take",
            preamble="texture <%(tp)s, 1, cudaReadModeElementType> tex_src;" % ctx)
    func = mod.get_function("take")
    tex_src = mod.get_texref("tex_src")
    func.prepare("PPI", (1,1,1), texrefs=[tex_src])
    return func, tex_src
            
@memoize
def get_copy_kernel(dtype_dest, dtype_src):
    return get_elwise_kernel(
            "%(tp_dest)s *dest, %(tp_src)s *src" % {
                "tp_dest": dtype_to_ctype(dtype_dest),
                "tp_src": dtype_to_ctype(dtype_src),
                },
            "dest[i] = src[i]",
            "copy")

@memoize
def get_axpbyz_kernel(dtype_x, dtype_y, dtype_z):
    return get_elwise_kernel(
            "%(tp_x)s a, %(tp_x)s *x, %(tp_y)s b, %(tp_y)s *y, %(tp_z)s *z" % {
                "tp_x": dtype_to_ctype(dtype_x),
                "tp_y": dtype_to_ctype(dtype_y),
                "tp_z": dtype_to_ctype(dtype_z),
                },
            "z[i] = a*x[i] + b*y[i]",
            "axpbyz")

@memoize
def get_axpbz_kernel(dtype):
    return get_elwise_kernel(
            "%(tp)s a, %(tp)s *x,%(tp)s b, %(tp)s *z" % {
                "tp": dtype_to_ctype(dtype)},
            "z[i] = a * x[i] + b",
            "axpb")

@memoize
def get_multiply_kernel(dtype_x, dtype_y, dtype_z):
    return get_elwise_kernel(
            "%(tp_x)s *x, %(tp_y)s *y, %(tp_z)s *z" % {
                "tp_x": dtype_to_ctype(dtype_x),
                "tp_y": dtype_to_ctype(dtype_y),
                "tp_z": dtype_to_ctype(dtype_z),
                },
            "z[i] = x[i] * y[i]",
            "multiply")

@memoize
def get_divide_kernel(dtype_x, dtype_y, dtype_z):
    return get_elwise_kernel(
            "%(tp_x)s *x, %(tp_y)s *y, %(tp_z)s *z" % {
                "tp_x": dtype_to_ctype(dtype_x),
                "tp_y": dtype_to_ctype(dtype_y),
                "tp_z": dtype_to_ctype(dtype_z),
                },
            "z[i] = x[i] / y[i]",
            "divide")

@memoize
def get_rdivide_elwise_kernel(dtype):
    return get_elwise_kernel(
            "%(tp)s *x, %(tp)s y, %(tp)s *z" % {
                "tp": dtype_to_ctype(dtype),
                },
            "z[i] = y / x[i]",
            "divide_r")



@memoize
def get_fill_kernel(dtype):
    return get_elwise_kernel(
            "%(tp)s a, %(tp)s *z" % {
                "tp": dtype_to_ctype(dtype),
                },
            "z[i] = a",
            "fill")

@memoize
def get_reverse_kernel(dtype):
    return get_elwise_kernel(
            "%(tp)s *y, %(tp)s *z" % {
                "tp": dtype_to_ctype(dtype),
                },
            "z[i] = y[n-1-i]",
            "reverse")

@memoize
def get_arange_kernel(dtype):
    return get_elwise_kernel(
            "%(tp)s *z, %(tp)s start, %(tp)s step" % {
                "tp": dtype_to_ctype(dtype),
                },
            "z[i] = start + i*step",
            "arange")


@memoize
def get_pow_kernel(dtype):
    if dtype == numpy.float64:
        func = "pow"
    else:
        func = "powf"

    return get_elwise_kernel(
            "%(tp)s value, %(tp)s *y, %(tp)s *z" % {
                "tp": dtype_to_ctype(dtype),
                },
            "z[i] = %s(y[i], value)" % func,
            "pow_method")

@memoize
def get_pow_array_kernel(dtype_x, dtype_y, dtype_z):
    if numpy.float64 in [dtype_x, dtype_y]:
        func = "pow"
    else:
        func = "powf"

    return get_elwise_kernel(
            "%(tp_x)s *x, %(tp_y)s *y, %(tp_z)s *z" % {
                "tp_x": dtype_to_ctype(dtype_x),
                "tp_y": dtype_to_ctype(dtype_y),
                "tp_z": dtype_to_ctype(dtype_z),
                },
            "z[i] = %s(x[i], y[i])" % func,
            "pow_method")

@memoize
def get_fmod_kernel():
    return get_elwise_kernel(
            "float *arg, float *mod, float *z",
            "z[i] = fmod(arg[i], mod[i])",
            "fmod_kernel")
    
@memoize
def get_modf_kernel():
    return get_elwise_kernel(
            "float *x, float *intpart ,float *fracpart",
            "fracpart[i] = modf(x[i], &intpart[i])",
            "modf_kernel")

@memoize
def get_frexp_kernel():
    return get_elwise_kernel(
            "float *x, float *significand, float *exponent",
            """
                int expt = 0;
                significand[i] = frexp(x[i], &expt);
                exponent[i] = expt;
            """,
            "frexp_kernel")

@memoize
def get_ldexp_kernel():
    return get_elwise_kernel(
            "float *sig, float *expt, float *z",
            "z[i] = ldexp(sig[i], int(expt[i]))",
            "ldexp_kernel")

@memoize
def get_unary_func_kernel(func_name, in_dtype, out_dtype=None):
    if out_dtype is None:
        out_dtype = in_dtype

    return get_elwise_kernel(
            "%(tp_in)s *y, %(tp_out)s *z" % {
                "tp_in": dtype_to_ctype(in_dtype),
                "tp_out": dtype_to_ctype(out_dtype),
                },
            "z[i] = %s(y[i])" % func_name,
            "%s_kernel" % func_name)    
