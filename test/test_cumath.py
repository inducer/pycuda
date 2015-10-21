from __future__ import division
from __future__ import absolute_import
import math
import numpy as np
from pycuda.tools import mark_cuda_test
from six.moves import range


def have_pycuda():
    try:
        import pycuda  # noqa
        return True
    except:
        return False


if have_pycuda():
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as drv  # noqa
    import pycuda.cumath as cumath


sizes = [10, 128, 1024, 1 << 10, 1 << 13]
dtypes = [np.float32, np.float64]
complex_dtypes = [np.complex64, np.complex128]


numpy_func_names = {
        "asin": "arcsin",
        "acos": "arccos",
        "atan": "arctan",
        }


def make_unary_function_test(name, a=0, b=1, threshold=0, complex=False):
    def test():
        gpu_func = getattr(cumath, name)
        cpu_func = getattr(np, numpy_func_names.get(name, name))
        if complex:
            _dtypes = complex_dtypes
        else:
            _dtypes = dtypes

        for s in sizes:
            for dtype in _dtypes:
                np.random.seed(1)
                A = (np.random.random(s)*(b-a) + a).astype(dtype)
                if complex:
                    A += (np.random.random(s)*(b-a) + a)*1j

                args = gpuarray.to_gpu(A)
                gpu_results = gpu_func(args).get()
                cpu_results = cpu_func(A)

                max_err = np.max(np.abs(cpu_results - gpu_results))
                assert (max_err <= threshold).all(), \
                        (max_err, name, dtype)

                gpu_results2 = gpuarray.empty_like(args)
                gr2 = gpu_func(args, out=gpu_results2)
                assert gpu_results2 is gr2
                gr2 = gr2.get()
                max_err = np.max(np.abs(cpu_results - gr2))
                assert (max_err <= threshold).all(), \
                        (max_err, name, dtype)

    return mark_cuda_test(test)


if have_pycuda():
    test_ceil = make_unary_function_test("ceil", -10, 10)
    test_floor = make_unary_function_test("ceil", -10, 10)
    test_fabs = make_unary_function_test("fabs", -10, 10)
    test_exp = make_unary_function_test("exp", -3, 3, 1e-5)
    test_exp_c = make_unary_function_test("exp", -3, 3, 1e-5, complex=True)
    test_log = make_unary_function_test("log", 1e-5, 1, 5e-7)
    test_log10 = make_unary_function_test("log10", 1e-5, 1, 3e-7)
    test_sqrt = make_unary_function_test("sqrt", 1e-5, 1, 2e-7)

    test_sin = make_unary_function_test("sin", -10, 10, 1e-7)
    test_sin_c = make_unary_function_test("sin", -3, 3, 2e-6, complex=True)
    test_cos = make_unary_function_test("cos", -10, 10, 1e-7)
    test_cos_c = make_unary_function_test("cos", -3, 3, 2e-6, complex=True)
    test_asin = make_unary_function_test("asin", -0.9, 0.9, 5e-7)
    #test_sin_c = make_unary_function_test("sin", -0.9, 0.9, 2e-6, complex=True)
    test_acos = make_unary_function_test("acos", -0.9, 0.9, 5e-7)
    #test_acos_c = make_unary_function_test("acos", -0.9, 0.9, 2e-6, complex=True)
    test_tan = make_unary_function_test("tan",
            -math.pi/2 + 0.1, math.pi/2 - 0.1, 1e-5)
    test_tan_c = make_unary_function_test("tan",
            -math.pi/2 + 0.1, math.pi/2 - 0.1, 3e-5, complex=True)
    test_atan = make_unary_function_test("atan", -10, 10, 2e-7)

    test_sinh = make_unary_function_test("sinh", -3, 3, 2e-6)
    test_sinh_c = make_unary_function_test("sinh", -3, 3, 3e-6, complex=True)
    test_cosh = make_unary_function_test("cosh", -3, 3, 2e-6)
    test_cosh_c = make_unary_function_test("cosh", -3, 3, 3e-6, complex=True)
    test_tanh = make_unary_function_test("tanh", -3, 3, 2e-6)
    test_tanh_c = make_unary_function_test("tanh",
            -math.pi/2 + 0.1, math.pi/2 - 0.1, 3e-5, complex=True)


class TestMath:
    disabled = not have_pycuda()

    @mark_cuda_test
    def test_fmod(self):
        """tests if the fmod function works"""
        for s in sizes:
            a = gpuarray.arange(s, dtype=np.float32)/10
            a2 = gpuarray.arange(s, dtype=np.float32)/45.2 + 0.1
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
            a = gpuarray.arange(s, dtype=np.float32)
            a2 = gpuarray.arange(s, dtype=np.float32)*1e-3
            b = cumath.ldexp(a, a2)

            a = a.get()
            a2 = a2.get()
            b = b.get()

            for i in range(s):
                assert math.ldexp(a[i], int(a2[i])) == b[i]

    @mark_cuda_test
    def test_modf(self):
        """tests if the modf function works"""
        for s in sizes:
            a = gpuarray.arange(s, dtype=np.float32)/10
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
            a = gpuarray.arange(s, dtype=np.float32)/10
            significands, exponents = cumath.frexp(a)

            a = a.get()
            significands = significands.get()
            exponents = exponents.get()

            for i in range(s):
                sig_true, ex_true = math.frexp(a[i])

                assert sig_true == significands[i]
                assert ex_true == exponents[i]

    @mark_cuda_test
    def test_unary_func_kwargs(self):
        """tests if the kwargs to the unary functions work"""
        from pycuda.driver import Stream

        name, a, b, threshold = ("exp", -3, 3, 1e-5)
        gpu_func = getattr(cumath, name)
        cpu_func = getattr(np, numpy_func_names.get(name, name))
        for s in sizes:
            for dtype in dtypes:
                np.random.seed(1)
                A = (np.random.random(s)*(b-a) + a).astype(dtype)
                if complex:
                    A = A + (np.random.random(s)*(b-a) + a)*1j

                np.random.seed(1)
                A = (np.random.random(s)*(b-a) + a).astype(dtype)
                args = gpuarray.to_gpu(A)

                # 'out' kw
                gpu_results = gpuarray.empty_like(args)
                gpu_results = gpu_func(args, out=gpu_results).get()
                cpu_results = cpu_func(A)
                max_err = np.max(np.abs(cpu_results - gpu_results))
                assert (max_err <= threshold).all(), (max_err, name, dtype)

                # 'out' position
                gpu_results = gpuarray.empty_like(args)
                gpu_results = gpu_func(args, gpu_results).get()
                cpu_results = cpu_func(A)
                max_err = np.max(np.abs(cpu_results - gpu_results))
                assert (max_err <= threshold).all(), (max_err, name, dtype)

                # 'stream' kw
                mystream = Stream()
                np.random.seed(1)
                A = (np.random.random(s)*(b-a) + a).astype(dtype)
                args = gpuarray.to_gpu(A)
                gpu_results = gpuarray.empty_like(args)
                gpu_results = gpu_func(args, stream=mystream).get()
                cpu_results = cpu_func(A)
                max_err = np.max(np.abs(cpu_results - gpu_results))
                assert (max_err <= threshold).all(), (max_err, name, dtype)

                # 'stream' position
                mystream = Stream()
                np.random.seed(1)
                A = (np.random.random(s)*(b-a) + a).astype(dtype)
                args = gpuarray.to_gpu(A)
                gpu_results = gpuarray.empty_like(args)
                gpu_results = gpu_func(args, mystream).get()
                cpu_results = cpu_func(A)
                max_err = np.max(np.abs(cpu_results - gpu_results))
                assert (max_err <= threshold).all(), (max_err, name, dtype)

                # 'out' and 'stream' kw
                mystream = Stream()
                np.random.seed(1)
                A = (np.random.random(s)*(b-a) + a).astype(dtype)
                args = gpuarray.to_gpu(A)
                gpu_results = gpuarray.empty_like(args)
                gpu_results = gpu_func(args, stream=mystream, out=gpu_results).get()
                cpu_results = cpu_func(A)
                max_err = np.max(np.abs(cpu_results - gpu_results))
                assert (max_err <= threshold).all(), (max_err, name, dtype)


if __name__ == "__main__":
    # make sure that import failures get reported, instead of skipping the tests.
    import pycuda.autoinit  # noqa

    import sys
    if len(sys.argv) > 1:
        exec (sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
