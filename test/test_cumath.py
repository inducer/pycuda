from __future__ import division
import math
import numpy
from pycuda.tools import mark_cuda_test

def have_pycuda():
    try:
        import pycuda
        return True
    except:
        return False


if have_pycuda():
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as drv
    import pycuda.cumath as cumath
    from pycuda.compiler import SourceModule




sizes = [10, 128, 1024, 1<<10, 1<<13]
dtypes = [numpy.float32, numpy.float64]



numpy_func_names = {
        "asin": "arcsin",
        "acos": "arccos",
        "atan": "arctan",
        }




def make_unary_function_test(name, (a, b)=(0, 1), threshold=0):
    def test():
        gpu_func = getattr(cumath, name)
        cpu_func = getattr(numpy, numpy_func_names.get(name, name))

        for s in sizes:
            for dtype in dtypes:
                args = gpuarray.arange(a, b, (b-a)/s, dtype=numpy.float32)
                gpu_results = gpu_func(args).get()
                cpu_results = cpu_func(args.get())

                max_err = numpy.max(numpy.abs(cpu_results - gpu_results))
                assert (max_err <= threshold).all(), \
                        (max_err, name, dtype)

    return mark_cuda_test(test)




if have_pycuda():
    test_ceil = make_unary_function_test("ceil", (-10, 10))
    test_floor = make_unary_function_test("ceil", (-10, 10))
    test_fabs = make_unary_function_test("fabs", (-10, 10))
    test_exp = make_unary_function_test("exp", (-3, 3), 1e-5)
    test_log = make_unary_function_test("log", (1e-5, 1), 5e-7)
    test_log10 = make_unary_function_test("log10", (1e-5, 1), 3e-7)
    test_sqrt = make_unary_function_test("sqrt", (1e-5, 1), 2e-7)

    test_sin = make_unary_function_test("sin", (-10, 10), 1e-7)
    test_cos = make_unary_function_test("cos", (-10, 10), 1e-7)
    test_asin = make_unary_function_test("asin", (-0.9, 0.9), 5e-7)
    test_acos = make_unary_function_test("acos", (-0.9, 0.9), 5e-7)
    test_tan = make_unary_function_test("tan", 
            (-math.pi/2 + 0.1, math.pi/2 - 0.1), 1e-5)
    test_atan = make_unary_function_test("atan", (-10, 10), 2e-7)

    test_sinh = make_unary_function_test("sinh", (-3, 3), 1e-6)
    test_cosh = make_unary_function_test("cosh", (-3, 3), 1e-6)
    test_tanh = make_unary_function_test("tanh", (-3, 3), 2e-6)




class TestMath:
    disabled = not have_pycuda()

    @mark_cuda_test
    def test_fmod(self):
        """tests if the fmod function works"""
        for s in sizes:
            a = gpuarray.arange(s, dtype=numpy.float32)/10
            a2 = gpuarray.arange(s, dtype=numpy.float32)/45.2 + 0.1
            b = cumath.fmod(a, a2)

            a = a.get()
            a2 = a2.get()
            b = b.get()

            for i in range(s):
                assert math.fmod(a[i], a2[i]) == b[i]

    @mark_cuda_test
    def test_ldexp(self):
        """tests if the ldexp function works"""
        for s in sizes:
            a = gpuarray.arange(s, dtype=numpy.float32)
            a2 = gpuarray.arange(s, dtype=numpy.float32)*1e-3
            b = cumath.ldexp(a,a2)

            a = a.get()
            a2 = a2.get()
            b = b.get()

            for i in range(s):
                assert math.ldexp(a[i], int(a2[i])) == b[i]

    @mark_cuda_test
    def test_modf(self):
        """tests if the modf function works"""
        for s in sizes:
            a = gpuarray.arange(s, dtype=numpy.float32)/10
            fracpart, intpart = cumath.modf(a)

            a = a.get()
            intpart = intpart.get()
            fracpart = fracpart.get()

            for i in range(s):
                fracpart_true, intpart_true = math.modf(a[i])

                assert intpart_true == intpart[i]
                assert abs(fracpart_true - fracpart[i]) < 1e-4

    @mark_cuda_test
    def test_frexp(self):
        """tests if the frexp function works"""
        for s in sizes:
            a = gpuarray.arange(s, dtype=numpy.float32)/10
            significands, exponents = cumath.frexp(a)

            a = a.get()
            significands = significands.get()
            exponents = exponents.get()

            for i in range(s):
                sig_true, ex_true = math.frexp(a[i])

                assert sig_true == significands[i]
                assert ex_true == exponents[i]

if __name__ == "__main__":
    # make sure that import failures get reported, instead of skipping the tests.
    import pycuda.autoinit

    import sys
    if len(sys.argv) > 1:
        exec sys.argv[1]
    else:
        from py.test.cmdline import main
        main([__file__])
