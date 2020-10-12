#!python 
import numpy
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.tools import context_dependent_memoize




def main(dtype):
    from pycuda.elementwise import get_linear_combination_kernel
    lc_kernel, lc_texrefs = get_linear_combination_kernel((
        (True, dtype, dtype),
        (True, dtype, dtype)
        ), dtype)

    for size_exp in range(10, 26):
        size = 1 << size_exp

        from pycuda.curandom import rand
        a = gpuarray.to_gpu(numpy.array(5, dtype=dtype))
        x = rand(size, dtype=dtype)
        b = gpuarray.to_gpu(numpy.array(7, dtype=dtype))
        y = rand(size, dtype=dtype)

        z = gpuarray.empty_like(x)

        start = drv.Event()
        stop = drv.Event()
        start.record()

        for i in range(20):
            a.bind_to_texref_ext(lc_texrefs[0], allow_double_hack=True)
            b.bind_to_texref_ext(lc_texrefs[1], allow_double_hack=True)
            lc_kernel.prepared_call(x._grid, x._block,
                x.gpudata, y.gpudata, z.gpudata, x.mem_size)

        stop.record()
        stop.synchronize()

        print(size, size_exp, stop.time_since(start))



@context_dependent_memoize
def get_lin_comb_kernel_no_tex(summand_descriptors,
        dtype_z):
    from pycuda.tools import dtype_to_ctype
    from pycuda.elementwise import \
            VectorArg, ScalarArg, get_elwise_module

    args = []
    loop_prep = []
    summands = []

    for i, (is_gpu_scalar, scalar_dtype, vector_dtype) in \
            enumerate(summand_descriptors):
        if is_gpu_scalar:
            args.append(VectorArg(vector_dtype, "global_a%d" % i))
            args.append(VectorArg(vector_dtype, "x%d" % i))
            loop_prep.append("%s a%d = *global_a%d"
                    % (dtype_to_ctype(scalar_dtype), i, i))
        else:
            args.append(ScalarArg(scalar_dtype, "a%d" % i))
            args.append(VectorArg(vector_dtype, "x%d" % i))

        summands.append("a%d*x%d[i]" % (i, i))

    args.append(VectorArg(dtype_z, "z"))
    args.append(ScalarArg(numpy.uintp, "n"))

    mod = get_elwise_module(args,
            "z[i] = " + " + ".join(summands),
            "linear_combination",
            loop_prep=";\n".join(loop_prep))

    func = mod.get_function("linear_combination")
    func.prepare("".join(arg.struct_char for arg in args))

    return func



def main_no_tex(dtype):
    lc_kernel = get_lin_comb_kernel_no_tex((
        (True, dtype, dtype),
        (True, dtype, dtype)
        ), dtype)

    for size_exp in range(10,26):
        size = 1 << size_exp

        from pycuda.curandom import rand
        a = gpuarray.to_gpu(numpy.array(5, dtype=dtype))
        x = rand(size, dtype=dtype)
        b = gpuarray.to_gpu(numpy.array(7, dtype=dtype))
        y = rand(size, dtype=dtype)

        z = gpuarray.empty_like(x)

        start = drv.Event()
        stop = drv.Event()
        start.record()

        for i in range(20):
            lc_kernel.prepared_call(x._grid, x._block,
                a.gpudata, x.gpudata,
                b.gpudata, y.gpudata,
                z.gpudata, x.mem_size)

        stop.record()
        stop.synchronize()

        print(size, size_exp, stop.time_since(start))




if __name__ == "__main__":
    dtype = numpy.float32

    main(dtype)
    print()
    main_no_tex(dtype)

