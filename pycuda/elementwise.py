"""Elementwise functionality."""

from __future__ import division

__copyright__ = "Copyright (C) 2009 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""




from pycuda.tools import context_dependent_memoize
import numpy as np
from pycuda.tools import dtype_to_ctype, VectorArg, ScalarArg
from pytools import memoize_method




def get_elwise_module(arguments, operation,
        name="kernel", keep=False, options=None,
        preamble="", loop_prep="", after_loop=""):
    from pycuda.compiler import SourceModule
    return SourceModule("""
        #include <pycuda-complex.hpp>

        %(preamble)s

        __global__ void %(name)s(%(arguments)s)
        {

          unsigned tid = threadIdx.x;
          unsigned total_threads = gridDim.x*blockDim.x;
          unsigned cta_start = blockDim.x*blockIdx.x;
          unsigned i;

          %(loop_prep)s;

          for (i = cta_start + tid; i < n; i += total_threads)
          {
            %(operation)s;
          }

          %(after_loop)s;
        }
        """ % {
            "arguments": ", ".join(arg.declarator() for arg in arguments),
            "operation": operation,
            "name": name,
            "preamble": preamble,
            "loop_prep": loop_prep,
            "after_loop": after_loop,
            },
        options=options, keep=keep)




def get_elwise_range_module(arguments, operation,
        name="kernel", keep=False, options=None,
        preamble="", loop_prep="", after_loop=""):
    from pycuda.compiler import SourceModule
    return SourceModule("""
        #include <pycuda-complex.hpp>

        %(preamble)s

        __global__ void %(name)s(%(arguments)s)
        {
          unsigned tid = threadIdx.x;
          unsigned total_threads = gridDim.x*blockDim.x;
          unsigned cta_start = blockDim.x*blockIdx.x;
          long i;

          %(loop_prep)s;

          if (step < 0)
          {
            for (i = start + (cta_start + tid)*step;
              i > stop; i += total_threads*step)
            {
              %(operation)s;
            }
          }
          else
          {
            for (i = start + (cta_start + tid)*step; 
              i < stop; i += total_threads*step)
            {
              %(operation)s;
            }
          }

          %(after_loop)s;
        }
        """ % {
            "arguments": ", ".join(arg.declarator() for arg in arguments),
            "operation": operation,
            "name": name,
            "preamble": preamble,
            "loop_prep": loop_prep,
            "after_loop": after_loop,
            },
        options=options, keep=keep)




def get_elwise_kernel_and_types(arguments, operation,
        name="kernel", keep=False, options=None, use_range=False, **kwargs):
    if isinstance(arguments, str):
        from pycuda.tools import parse_c_arg
        arguments = [parse_c_arg(arg) for arg in arguments.split(",")]

    if use_range:
        arguments.extend([
            ScalarArg(np.intp, "start"),
            ScalarArg(np.intp, "stop"),
            ScalarArg(np.intp, "step"),
            ])
    else:
        arguments.append(ScalarArg(np.uintp, "n"))

    if use_range:
        module_builder = get_elwise_range_module
    else:
        module_builder = get_elwise_module

    mod = module_builder(arguments, operation, name,
            keep, options, **kwargs)

    from pycuda.tools import get_arg_type
    func = mod.get_function(name)
    func.prepare("".join(arg.struct_char for arg in arguments))

    return func, arguments

def get_elwise_kernel(arguments, operation,
        name="kernel", keep=False, options=None, **kwargs):
    """Return a L{pycuda.driver.Function} that performs the same scalar operation
    on one or several vectors.
    """
    func, arguments = get_elwise_kernel_and_types(
            arguments, operation, name, keep, options, **kwargs)

    return func




class ElementwiseKernel:
    def __init__(self, arguments, operation,
            name="kernel", keep=False, options=None, **kwargs):

        self.gen_kwargs = kwargs.copy()
        self.gen_kwargs.update(dict(keep=keep, options=options, name=name,
            operation=operation, arguments=arguments))

    @memoize_method
    def generate_stride_kernel_and_types(self, use_range):
        knl, arguments = get_elwise_kernel_and_types(use_range=use_range, 
                **self.gen_kwargs)

        assert [i for i, arg in enumerate(arguments)
                if isinstance(arg, VectorArg)], \
                "ElementwiseKernel can only be used with functions that have at least one " \
                "vector argument"

        return knl, arguments

    def __call__(self, *args, **kwargs):
        vectors = []

        range_ = kwargs.pop("range", None)
        slice_ = kwargs.pop("slice", None)
        stream = kwargs.pop("stream", None)

        if kwargs:
            raise TypeError("invalid keyword arguments specified: "
                    + ", ".join(kwargs.iterkeys()))

        invocation_args = []
        func, arguments = self.generate_stride_kernel_and_types(
                range_ is not None or slice_ is not None)

        for arg, arg_descr in zip(args, arguments):
            if isinstance(arg_descr, VectorArg):
                if not arg.flags.forc:
                    raise RuntimeError("elementwise kernel cannot "
                            "deal with non-contiguous arrays")

                vectors.append(arg)
                invocation_args.append(arg.gpudata)
            else:
                invocation_args.append(arg)

        repr_vec = vectors[0]

        if slice_ is not None:
            if range_ is not None:
                raise TypeError("may not specify both range and slice "
                        "keyword arguments")

            range_ = slice(*slice_.indices(repr_vec.size))

        if range_ is not None:
            invocation_args.append(range_.start)
            invocation_args.append(range_.stop)
            if range_.step is None:
                invocation_args.append(1)
            else:
                invocation_args.append(range_.step)

            from pycuda.gpuarray import splay
            grid, block = splay(abs(range_.stop - range_.start)//range_.step)
        else:
            block = repr_vec._block
            grid = repr_vec._grid
            invocation_args.append(repr_vec.mem_size)

        func.prepared_async_call(grid, block, stream, *invocation_args)




@context_dependent_memoize
def get_take_kernel(dtype, idx_dtype, vec_count=1):
    ctx = {
            "idx_tp": dtype_to_ctype(idx_dtype),
            "tp": dtype_to_ctype(dtype),
            "tex_tp": dtype_to_ctype(dtype, with_fp_tex_hack=True),
            }

    args = [VectorArg(idx_dtype, "idx")] + [
            VectorArg(dtype, "dest"+str(i))for i in range(vec_count)] + [
            ScalarArg(np.intp, "n")]
    preamble = "#include <pycuda-helpers.hpp>\n\n" + "\n".join(
        "texture <%s, 1, cudaReadModeElementType> tex_src%d;" % (ctx["tex_tp"], i)
        for i in range(vec_count))
    body = (
            ("%(idx_tp)s src_idx = idx[i];\n" % ctx)
            + "\n".join(
            "dest%d[i] = fp_tex1Dfetch(tex_src%d, src_idx);" % (i, i)
            for i in range(vec_count)))

    mod = get_elwise_module(args, body, "take", preamble=preamble)
    func = mod.get_function("take")
    tex_src = [mod.get_texref("tex_src%d" % i) for i in range(vec_count)]
    func.prepare("P"+(vec_count*"P")+np.dtype(np.uintp).char, texrefs=tex_src)
    return func, tex_src




@context_dependent_memoize
def get_take_put_kernel(dtype, idx_dtype, with_offsets, vec_count=1):
    ctx = {
            "idx_tp": dtype_to_ctype(idx_dtype),
            "tp": dtype_to_ctype(dtype),
            "tex_tp": dtype_to_ctype(dtype, with_fp_tex_hack=True),
            }

    args = [
            VectorArg(idx_dtype, "gmem_dest_idx"),
            VectorArg(idx_dtype, "gmem_src_idx"),
            ] + [
            VectorArg(dtype, "dest%d" % i)
                for i in range(vec_count)
            ] + [
            ScalarArg(idx_dtype, "offset%d" % i)
                for i in range(vec_count) if with_offsets
            ] + [ScalarArg(np.intp, "n")]

    preamble = "#include <pycuda-helpers.hpp>\n\n" + "\n".join(
        "texture <%s, 1, cudaReadModeElementType> tex_src%d;" % (ctx["tex_tp"], i)
        for i in range(vec_count))

    if with_offsets:
        def get_copy_insn(i):
            return ("dest%d[dest_idx] = "
                    "fp_tex1Dfetch(tex_src%d, src_idx+offset%d);"
                    % (i, i, i))
    else:
        def get_copy_insn(i):
            return ("dest%d[dest_idx] = "
                    "fp_tex1Dfetch(tex_src%d, src_idx);" % (i, i))

    body = (("%(idx_tp)s src_idx = gmem_src_idx[i];\n"
                "%(idx_tp)s dest_idx = gmem_dest_idx[i];\n" % ctx)
            + "\n".join(get_copy_insn(i) for i in range(vec_count)))

    mod = get_elwise_module(args, body, "take_put", preamble=preamble)
    func = mod.get_function("take_put")
    tex_src = [mod.get_texref("tex_src%d" % i) for i in range(vec_count)]

    func.prepare(
            "PP"+(vec_count*"P")
            +(bool(with_offsets)*vec_count*idx_dtype.char)
            +np.dtype(np.uintp).char,
            texrefs=tex_src)
    return func, tex_src




@context_dependent_memoize
def get_put_kernel(dtype, idx_dtype, vec_count=1):
    ctx = {
            "idx_tp": dtype_to_ctype(idx_dtype),
            "tp": dtype_to_ctype(dtype),
            }

    args = [
            VectorArg(idx_dtype, "gmem_dest_idx"),
            ] + [
            VectorArg(dtype, "dest%d" % i)
                for i in range(vec_count)
            ] + [
            VectorArg(dtype, "src%d" % i)
                for i in range(vec_count)
            ] + [ScalarArg(np.intp, "n")]

    body = (
            "%(idx_tp)s dest_idx = gmem_dest_idx[i];\n" % ctx
            + "\n".join("dest%d[dest_idx] = src%d[i];" % (i, i)
                for i in range(vec_count)))

    func = get_elwise_module(args, body, "put").get_function("put")
    func.prepare("P"+(2*vec_count*"P")+np.dtype(np.uintp).char)
    return func




@context_dependent_memoize
def get_copy_kernel(dtype_dest, dtype_src):
    return get_elwise_kernel(
            "%(tp_dest)s *dest, %(tp_src)s *src" % {
                "tp_dest": dtype_to_ctype(dtype_dest),
                "tp_src": dtype_to_ctype(dtype_src),
                },
            "dest[i] = src[i]",
            "copy")



@context_dependent_memoize
def get_linear_combination_kernel(summand_descriptors,
        dtype_z):
    from pycuda.tools import dtype_to_ctype
    from pycuda.elementwise import \
            VectorArg, ScalarArg, get_elwise_module

    args = []
    preamble = [ "#include <pycuda-helpers.hpp>\n\n" ]
    loop_prep = []
    summands = []
    tex_names = []

    for i, (is_gpu_scalar, scalar_dtype, vector_dtype) in \
            enumerate(summand_descriptors):
        if is_gpu_scalar:
            preamble.append(
                    "texture <%s, 1, cudaReadModeElementType> tex_a%d;"
                    % (dtype_to_ctype(scalar_dtype, with_fp_tex_hack=True), i))
            args.append(VectorArg(vector_dtype, "x%d" % i))
            tex_names.append("tex_a%d" % i)
            loop_prep.append(
                    "%s a%d = fp_tex1Dfetch(tex_a%d, 0)"
                    % (dtype_to_ctype(scalar_dtype), i, i))
        else:
            args.append(ScalarArg(scalar_dtype, "a%d" % i))
            args.append(VectorArg(vector_dtype, "x%d" % i))

        summands.append("a%d*x%d[i]" % (i, i))

    args.append(VectorArg(dtype_z, "z"))
    args.append(ScalarArg(np.uintp, "n"))

    mod = get_elwise_module(args,
            "z[i] = " + " + ".join(summands),
            "linear_combination",
            preamble="\n".join(preamble),
            loop_prep=";\n".join(loop_prep))

    func = mod.get_function("linear_combination")
    tex_src = [mod.get_texref(tn) for tn in tex_names]
    func.prepare("".join(arg.struct_char for arg in args),
            texrefs=tex_src)

    return func, tex_src




@context_dependent_memoize
def get_axpbyz_kernel(dtype_x, dtype_y, dtype_z):
    return get_elwise_kernel(
            "%(tp_x)s a, %(tp_x)s *x, %(tp_y)s b, %(tp_y)s *y, %(tp_z)s *z" % {
                "tp_x": dtype_to_ctype(dtype_x),
                "tp_y": dtype_to_ctype(dtype_y),
                "tp_z": dtype_to_ctype(dtype_z),
                },
            "z[i] = a*x[i] + b*y[i]",
            "axpbyz")

@context_dependent_memoize
def get_axpbz_kernel(dtype):
    return get_elwise_kernel(
            "%(tp)s a, %(tp)s *x,%(tp)s b, %(tp)s *z" % {
                "tp": dtype_to_ctype(dtype)},
            "z[i] = a * x[i] + b",
            "axpb")

@context_dependent_memoize
def get_binary_op_kernel(dtype_x, dtype_y, dtype_z, operator):
    return get_elwise_kernel(
            "%(tp_x)s *x, %(tp_y)s *y, %(tp_z)s *z" % {
                "tp_x": dtype_to_ctype(dtype_x),
                "tp_y": dtype_to_ctype(dtype_y),
                "tp_z": dtype_to_ctype(dtype_z),
                },
            "z[i] = x[i] %s y[i]" % operator,
            "multiply")

@context_dependent_memoize
def get_rdivide_elwise_kernel(dtype):
    return get_elwise_kernel(
            "%(tp)s *x, %(tp)s y, %(tp)s *z" % {
                "tp": dtype_to_ctype(dtype),
                },
            "z[i] = y / x[i]",
            "divide_r")

@context_dependent_memoize
def get_binary_func_kernel(func, dtype_x, dtype_y, dtype_z):
    return get_elwise_kernel(
            "%(tp_x)s *x, %(tp_y)s *y, %(tp_z)s *z" % {
                "tp_x": dtype_to_ctype(dtype_x),
                "tp_y": dtype_to_ctype(dtype_y),
                "tp_z": dtype_to_ctype(dtype_z),
                },
            "z[i] = %s(x[i], y[i])" % func,
            func+"_kernel")

def get_binary_minmax_kernel(func, dtype_x, dtype_y, dtype_z):
    if not np.float64 in [dtype_x, dtype_y]:
        func = func +"f"

    from pytools import any
    if any(dt.kind == "f" for dt in [dtype_x, dtype_y, dtype_z]):
        func = "f"+func

    return get_binary_func_kernel(func, dtype_x, dtype_y, dtype_z)

@context_dependent_memoize
def get_fill_kernel(dtype):
    return get_elwise_kernel(
            "%(tp)s a, %(tp)s *z" % {
                "tp": dtype_to_ctype(dtype),
                },
            "z[i] = a",
            "fill")

@context_dependent_memoize
def get_reverse_kernel(dtype):
    return get_elwise_kernel(
            "%(tp)s *y, %(tp)s *z" % {
                "tp": dtype_to_ctype(dtype),
                },
            "z[i] = y[n-1-i]",
            "reverse")

@context_dependent_memoize
def get_real_kernel(dtype, real_dtype):
    return get_elwise_kernel(
            "%(tp)s *y, %(real_tp)s *z" % {
                "tp": dtype_to_ctype(dtype),
                "real_tp": dtype_to_ctype(real_dtype),
                },
            "z[i] = real(y[i])",
            "real")

@context_dependent_memoize
def get_imag_kernel(dtype, real_dtype):
    return get_elwise_kernel(
            "%(tp)s *y, %(real_tp)s *z" % {
                "tp": dtype_to_ctype(dtype),
                "real_tp": dtype_to_ctype(real_dtype),
                },
            "z[i] = imag(y[i])",
            "imag")

@context_dependent_memoize
def get_conj_kernel(dtype):
    return get_elwise_kernel(
            "%(tp)s *y, %(tp)s *z" % {
                "tp": dtype_to_ctype(dtype),
                },
            "z[i] = pycuda::conj(y[i])",
            "conj")

@context_dependent_memoize
def get_arange_kernel(dtype):
    return get_elwise_kernel(
            "%(tp)s *z, %(tp)s start, %(tp)s step" % {
                "tp": dtype_to_ctype(dtype),
                },
            "z[i] = start + i*step",
            "arange")


@context_dependent_memoize
def get_pow_kernel(dtype):
    if dtype == np.float64:
        func = "pow"
    else:
        func = "powf"

    return get_elwise_kernel(
            "%(tp)s value, %(tp)s *y, %(tp)s *z" % {
                "tp": dtype_to_ctype(dtype),
                },
            "z[i] = %s(y[i], value)" % func,
            "pow_method")

@context_dependent_memoize
def get_pow_array_kernel(dtype_x, dtype_y, dtype_z):
    if np.float64 in [dtype_x, dtype_y]:
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

@context_dependent_memoize
def get_fmod_kernel():
    return get_elwise_kernel(
            "float *arg, float *mod, float *z",
            "z[i] = fmod(arg[i], mod[i])",
            "fmod_kernel")

@context_dependent_memoize
def get_modf_kernel():
    return get_elwise_kernel(
            "float *x, float *intpart ,float *fracpart",
            "fracpart[i] = modf(x[i], &intpart[i])",
            "modf_kernel")

@context_dependent_memoize
def get_frexp_kernel():
    return get_elwise_kernel(
            "float *x, float *significand, float *exponent",
            """
                int expt = 0;
                significand[i] = frexp(x[i], &expt);
                exponent[i] = expt;
            """,
            "frexp_kernel")

@context_dependent_memoize
def get_ldexp_kernel():
    return get_elwise_kernel(
            "float *sig, float *expt, float *z",
            "z[i] = ldexp(sig[i], int(expt[i]))",
            "ldexp_kernel")

@context_dependent_memoize
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




@context_dependent_memoize
def get_if_positive_kernel(crit_dtype, dtype):
    return get_elwise_kernel([
            VectorArg(crit_dtype, "crit"),
            VectorArg(dtype, "then_"),
            VectorArg(dtype, "else_"),
            VectorArg(dtype, "result"),
            ],
            "result[i] = crit[i] > 0 ? then_[i] : else_[i]",
            "if_positive")
