#! /usr/bin/env python

import numpy as np
import numpy.linalg as la
import sys
from pycuda.tools import init_cuda_context_fixture
from pycuda.characterize import has_double_support


import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pytest
import operator


@pytest.fixture(autouse=True)
def init_cuda_context():
    yield from init_cuda_context_fixture()


def get_random_array(rng, shape, dtype):
    dtype = np.dtype(dtype)

    if dtype.kind == "f":
        return rng.random(shape, dtype)
    elif dtype.kind in "il":
        return rng.integers(-42, 42, shape, dtype)
    elif dtype.kind in "u":
        return rng.integers(0, 42, shape, dtype)
    elif dtype.kind == "c":
        real_dtype = np.empty(0, dtype).real.dtype
        return (dtype.type(1j) * get_random_array(rng, shape, real_dtype)
                + get_random_array(rng, shape, real_dtype))
    else:
        raise NotImplementedError(f"dtype = {dtype}")


def skip_if_not_enough_gpu_memory(required_mem_gigabytes):
    device_mem_GB = drv.Context.get_device().total_memory() / 1e9
    if device_mem_GB < required_mem_gigabytes:
        pytest.skip("Need at least %.1f GB memory" % required_mem_gigabytes)


@pytest.mark.cuda
class TestGPUArray:
    def test_pow_array(self):
        a = np.array([1, 2, 3, 4, 5]).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)
        b = np.array([1, 2, 3, 4, 5]).astype(np.float64)
        b_gpu = gpuarray.to_gpu(b)

        result = pow(a_gpu, b_gpu).get()
        np.testing.assert_allclose(a ** b, result, rtol=1e-6)

        result = (a_gpu ** b_gpu).get()
        np.testing.assert_allclose(pow(a, b), result, rtol=1e-6)

        a_gpu **= b_gpu
        a_gpu = a_gpu.get()
        np.testing.assert_allclose(pow(a, b), a_gpu, rtol=1e-6)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_pow_number(self, dtype):
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(dtype)
        a_gpu = gpuarray.to_gpu(a)

        result = pow(a_gpu, 2).get()
        np.testing.assert_allclose(a ** 2, result, rtol=1e-6)

        a_gpu **= 2
        a_gpu = a_gpu.get()
        np.testing.assert_allclose(a ** 2, a_gpu, rtol=1e-6)

    def test_rpow_array(self):
        scalar = np.random.rand()
        a = abs(np.random.rand(10))
        a_gpu = gpuarray.to_gpu(a)

        result = (scalar ** a_gpu).get()
        np.testing.assert_allclose(scalar ** a, result)

        result = (a_gpu ** a_gpu).get()
        np.testing.assert_allclose(a ** a, result)

        result = (a_gpu ** scalar).get()
        np.testing.assert_allclose(a ** scalar, result)

    def test_numpy_integer_shape(self):
        gpuarray.empty(np.int32(17), np.float32)
        gpuarray.empty((np.int32(17), np.int32(17)), np.float32)

    def test_ndarray_shape(self):
        gpuarray.empty(np.array(3), np.float32)
        gpuarray.empty(np.array([3]), np.float32)
        gpuarray.empty(np.array([2, 3]), np.float32)

    def test_abs(self):
        a = -gpuarray.arange(111, dtype=np.float32)
        res = a.get()

        for i in range(111):
            assert res[i] <= 0

        a = abs(a)

        res = a.get()

        for i in range(111):
            assert abs(res[i]) >= 0
            assert res[i] == i

    def test_len(self):
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
        a_cpu = gpuarray.to_gpu(a)
        assert len(a_cpu) == 10

    def test_multiply(self):
        """Test the muliplication of an array with a scalar. """

        for sz in [10, 50000]:
            for dtype, scalars in [(np.float32, [2]), (np.complex64, [2, 2j])]:
                for scalar in scalars:
                    a = np.arange(sz).astype(dtype)
                    a_gpu = gpuarray.to_gpu(a)
                    a_doubled = (scalar * a_gpu).get()

                    assert (a * scalar == a_doubled).all()

    def test_rmul_yields_right_type(self):
        a = np.array([1, 2, 3, 4, 5]).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)

        two_a = 2 * a_gpu
        assert isinstance(two_a, gpuarray.GPUArray)

        two_a = np.float32(2) * a_gpu
        assert isinstance(two_a, gpuarray.GPUArray)

    def test_multiply_array(self):
        """Test the multiplication of two arrays."""

        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
        b = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]).astype(np.float32)
        c = np.array(2)

        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = gpuarray.to_gpu(c)

        a_mul_b = (a_gpu * b_gpu).get()
        assert (a * b == a_mul_b).all()

        b_mul_a = (b_gpu * a_gpu).get()
        assert (b * a == b_mul_a).all()

        a_mul_c = (a_gpu * c_gpu).get()
        assert (a * c == a_mul_c).all()

        b_mul_c = (b_gpu * c_gpu).get()
        assert (b * c == b_mul_c).all()

    def test_unit_multiply_array(self):

        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)

        a_gpu = gpuarray.to_gpu(a)
        np.testing.assert_allclose(+a_gpu.get(), +a, rtol=1e-6)
        np.testing.assert_allclose(-a_gpu.get(), -a, rtol=1e-6)

    def test_addition_array(self):
        """Test the addition of two arrays."""

        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)
        b = np.array(1).astype(np.float32)
        b_gpu = gpuarray.to_gpu(b)
        a_added = (a_gpu + a_gpu).get()
        a_added_scalar = (a_gpu + 1).get()
        scalar_added_a = (1 + a_gpu).get()
        a_gpu_pl_b_gpu = (a_gpu + b_gpu).get()
        b_gpu_pl_a_gpu = (b_gpu + a_gpu).get()

        assert (a + a == a_added).all()
        assert (a + 1 == a_added_scalar).all()
        assert (1 + a == scalar_added_a).all()
        assert (a + b == a_gpu_pl_b_gpu).all()
        assert (b + a == b_gpu_pl_a_gpu).all()

    def test_iaddition_array(self):
        """Test the inplace addition of two arrays."""

        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)
        a_gpu += a_gpu
        a_added = a_gpu.get()

        assert (a + a == a_added).all()

    def test_addition_scalar(self):
        """Test the addition of an array and a scalar."""

        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)
        a_added = (7 + a_gpu).get()

        assert (7 + a == a_added).all()

    def test_iaddition_scalar(self):
        """Test the inplace addition of an array and a scalar."""

        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)
        a_gpu += 7
        a_added = a_gpu.get()

        assert (7 + a == a_added).all()

    def test_substract_array(self):
        """Test the subtraction of two arrays."""
        # test data
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
        b = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]).astype(np.float32)
        c = np.array(1).astype(np.float32)

        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = gpuarray.to_gpu(c)

        result = (a_gpu - b_gpu).get()
        assert (a - b == result).all()

        result = (b_gpu - a_gpu).get()
        assert (b - a == result).all()

        result = (a_gpu - c_gpu).get()
        assert (a - c == result).all()

        result = (c_gpu - a_gpu).get()
        assert (c - a == result).all()

    def test_substract_scalar(self):
        """Test the subtraction of an array and a scalar."""

        # test data
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)

        # convert a to a gpu object
        a_gpu = gpuarray.to_gpu(a)

        result = (a_gpu - 7).get()
        assert (a - 7 == result).all()

        result = (7 - a_gpu).get()
        assert (7 - a == result).all()

    def test_divide_scalar(self):
        """Test the division of an array and a scalar."""

        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)

        result = (a_gpu / 2).get()
        assert (a / 2 == result).all()

        result = (2 / a_gpu).get()
        assert (2 / a == result).all()

    def test_divide_array(self):
        """Test the division of an array and a scalar. """

        # test data
        a = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]).astype(np.float32)
        b = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10]).astype(np.float32)
        c = np.array(2)

        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = gpuarray.to_gpu(c)

        a_divide = (a_gpu / b_gpu).get()
        assert (np.abs(a / b - a_divide) < 1e-3).all()

        a_divide = (b_gpu / a_gpu).get()
        assert (np.abs(b / a - a_divide) < 1e-3).all()

        a_divide = (a_gpu / c_gpu).get()
        assert (np.abs(a / c - a_divide) < 1e-3).all()

        a_divide = (c_gpu / a_gpu).get()
        assert (np.abs(c / a - a_divide) < 1e-3).all()

    def test_random(self):
        from pycuda.curandom import rand as curand

        if has_double_support():
            dtypes = [np.float32, np.float64]
        else:
            dtypes = [np.float32]

        for dtype in dtypes:
            a = curand((10, 100), dtype=dtype).get()

            assert (0 <= a).all()
            assert (a < 1).all()

    def test_curand_wrappers(self):
        from pycuda.curandom import get_curand_version

        if get_curand_version() is None:
            from pytest import skip

            skip("curand not installed")

        generator_types = []
        if get_curand_version() >= (3, 2, 0):
            from pycuda.curandom import (
                XORWOWRandomNumberGenerator,
                Sobol32RandomNumberGenerator,
            )

            generator_types.extend(
                [XORWOWRandomNumberGenerator, Sobol32RandomNumberGenerator]
            )
        if get_curand_version() >= (4, 0, 0):
            from pycuda.curandom import (
                ScrambledSobol32RandomNumberGenerator,
                Sobol64RandomNumberGenerator,
                ScrambledSobol64RandomNumberGenerator,
            )

            generator_types.extend(
                [
                    ScrambledSobol32RandomNumberGenerator,
                    Sobol64RandomNumberGenerator,
                    ScrambledSobol64RandomNumberGenerator,
                ]
            )
        if get_curand_version() >= (4, 1, 0):
            from pycuda.curandom import MRG32k3aRandomNumberGenerator

            generator_types.extend([MRG32k3aRandomNumberGenerator])

        if has_double_support():
            dtypes = [np.float32, np.float64]
        else:
            dtypes = [np.float32]

        for gen_type in generator_types:
            gen = gen_type()

            for dtype in dtypes:
                gen.gen_normal(10000, dtype)
                # test non-Box-Muller version, if available
                gen.gen_normal(10001, dtype)

                if get_curand_version() >= (4, 0, 0):
                    gen.gen_log_normal(10000, dtype, 10.0, 3.0)
                    # test non-Box-Muller version, if available
                    gen.gen_log_normal(10001, dtype, 10.0, 3.0)

                x = gen.gen_uniform(10000, dtype)
                x_host = x.get()
                assert (-1 <= x_host).all()
                assert (x_host <= 1).all()

            gen.gen_uniform(10000, np.uint32)
            if get_curand_version() >= (5, 0, 0):
                gen.gen_poisson(10000, np.uint32, 13.0)
                for dtype in dtypes + [np.uint32]:
                    a = gpuarray.empty(1000000, dtype=dtype)
                    v = 10
                    a.fill(v)
                    gen.fill_poisson(a)
                    tmp = (a.get() == (v - 1)).sum() / a.size  # noqa: F841
                    # Commented out for CI on the off chance it'd fail
                    # # Check Poisson statistics (need 1e6 values)
                    # # Compare with scipy.stats.poisson.pmf(v - 1, v)
                    # assert np.isclose(0.12511, tmp, atol=0.002)

    def test_array_gt(self):
        """Test whether array contents are > the other array's
        contents"""

        a = np.array([5, 10]).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)
        b = np.array([2, 10]).astype(np.float32)
        b_gpu = gpuarray.to_gpu(b)
        result = (a_gpu > b_gpu).get()
        assert result[0]
        assert not result[1]

    def test_array_lt(self):
        """Test whether array contents are < the other array's
        contents"""

        a = np.array([5, 10]).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)
        b = np.array([2, 10]).astype(np.float32)
        b_gpu = gpuarray.to_gpu(b)
        result = (b_gpu < a_gpu).get()
        assert result[0]
        assert not result[1]

    def test_array_le(self):
        """Test whether array contents are <= the other array's
        contents"""

        a = np.array([5, 10, 1]).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)
        b = np.array([2, 10, 2]).astype(np.float32)
        b_gpu = gpuarray.to_gpu(b)
        result = (b_gpu <= a_gpu).get()
        assert result[0]
        assert result[1]
        assert not result[2]

    def test_array_ge(self):
        """Test whether array contents are >= the other array's
        contents"""

        a = np.array([5, 10, 1]).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)
        b = np.array([2, 10, 2]).astype(np.float32)
        b_gpu = gpuarray.to_gpu(b)
        result = (a_gpu >= b_gpu).get()
        assert result[0]
        assert result[1]
        assert not result[2]

    def test_array_eq(self):
        """Test whether array contents are == the other array's
        contents"""

        a = np.array([5, 10]).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)
        b = np.array([2, 10]).astype(np.float32)
        b_gpu = gpuarray.to_gpu(b)
        result = (a_gpu == b_gpu).get()
        assert not result[0]
        assert result[1]

    def test_array_ne(self):
        """Test whether array contents are != the other array's
        contents"""

        a = np.array([5, 10]).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)
        b = np.array([2, 10]).astype(np.float32)
        b_gpu = gpuarray.to_gpu(b)
        result = (a_gpu != b_gpu).get()
        assert result[0]
        assert not result[1]

    def test_nan_arithmetic(self):
        def make_nan_contaminated_vector(size):
            shape = (size,)
            a = np.random.randn(*shape).astype(np.float32)
            # for i in range(0, shape[0], 3):
            # a[i] = float('nan')
            from random import randrange

            for i in range(size // 10):
                a[randrange(0, size)] = float("nan")
            return a

        size = 1 << 20

        a = make_nan_contaminated_vector(size)
        a_gpu = gpuarray.to_gpu(a)
        b = make_nan_contaminated_vector(size)
        b_gpu = gpuarray.to_gpu(b)

        ab = a * b
        ab_gpu = (a_gpu * b_gpu).get()

        assert (np.isnan(ab) == np.isnan(ab_gpu)).all()

    def test_elwise_kernel(self):
        from pycuda.curandom import rand as curand

        a_gpu = curand((50,))
        b_gpu = curand((50,))

        from pycuda.elementwise import ElementwiseKernel

        lin_comb = ElementwiseKernel(
            "float a, float *x, float b, float *y, float *z",
            "z[i] = a*x[i] + b*y[i]",
            "linear_combination",
        )

        c_gpu = gpuarray.empty_like(a_gpu)
        lin_comb(5, a_gpu, 6, b_gpu, c_gpu)

        assert la.norm((c_gpu - (5 * a_gpu + 6 * b_gpu)).get()) < 1e-5

    def test_ranged_elwise_kernel(self):
        from pycuda.elementwise import ElementwiseKernel

        set_to_seven = ElementwiseKernel("float *z", "z[i] = 7", "set_to_seven")

        for i, slc in enumerate(
            [
                slice(5, 20000),
                slice(5, 20000, 17),
                slice(3000, 5, -1),
                slice(1000, -1),
            ]
        ):

            a_gpu = gpuarray.zeros((50000,), dtype=np.float32)
            a_cpu = np.zeros(a_gpu.shape, a_gpu.dtype)

            a_cpu[slc] = 7
            set_to_seven(a_gpu, slice=slc)
            drv.Context.synchronize()

            assert la.norm(a_cpu - a_gpu.get()) == 0, i

    def test_take(self):
        idx = gpuarray.arange(0, 10000, 2, dtype=np.uint32)
        for dtype in [np.float32, np.complex64]:
            a = gpuarray.arange(0, 600000, dtype=np.uint32).astype(dtype)
            a_host = a.get()
            result = gpuarray.take(a, idx)

            assert (a_host[idx.get()] == result.get()).all()

    def test_arange(self):
        a = gpuarray.arange(12, dtype=np.float32)
        assert (np.arange(12, dtype=np.float32) == a.get()).all()

    def test_ones(self):

        ones = np.ones(10)
        ones_gpu = gpuarray.ones(10)

        np.testing.assert_allclose(ones, ones_gpu.get(), rtol=1e-6)
        assert ones.dtype == ones_gpu.dtype

    @pytest.mark.parametrize("order", ["F", "C"])
    @pytest.mark.parametrize("input_dims", [0, 1, 2])
    def test_stack(self, order, input_dims):

        shape = (2, 2, 2)[:input_dims]
        axis = -1 if order == "F" else 0

        from numpy.random import default_rng
        rng = default_rng()
        x_in = rng.random(size=shape)
        y_in = rng.random(size=shape)
        x_in = x_in if order == "C" else np.asfortranarray(x_in)
        y_in = y_in if order == "C" else np.asfortranarray(y_in)

        x_gpu = gpuarray.to_gpu(x_in)
        y_gpu = gpuarray.to_gpu(y_in)

        numpy_stack = np.stack((x_in, y_in), axis=axis)
        gpuarray_stack = gpuarray.stack((x_gpu, y_gpu), axis=axis)

        np.testing.assert_allclose(gpuarray_stack.get(), numpy_stack)

        assert gpuarray_stack.shape == numpy_stack.shape

    def test_concatenate(self):

        from pycuda.curandom import rand as curand

        a_dev = curand((5, 15, 20), dtype=np.float32)
        b_dev = curand((4, 15, 20), dtype=np.float32)
        c_dev = curand((3, 15, 20), dtype=np.float32)
        a = a_dev.get()
        b = b_dev.get()
        c = c_dev.get()

        cat_dev = gpuarray.concatenate((a_dev, b_dev, c_dev))
        cat = np.concatenate((a, b, c))

        np.testing.assert_allclose(cat, cat_dev.get())

        assert cat.shape == cat_dev.shape

    def test_reverse(self):
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
        a_cpu = gpuarray.to_gpu(a)

        a_cpu = a_cpu.reverse()

        b = a_cpu.get()

        for i in range(0, 10):
            assert a[len(a) - 1 - i] == b[i]

    def test_sum(self):
        from pycuda.curandom import rand as curand

        a_gpu = curand((200000,))
        a = a_gpu.get()

        sum_a = np.sum(a)

        sum_a_gpu = gpuarray.sum(a_gpu).get()

        assert abs(sum_a_gpu - sum_a) / abs(sum_a) < 1e-4

    @pytest.mark.parametrize("dtype", [np.int32, np.bool, np.float32, np.float64])
    def test_any(self, dtype):

        ary_list = [np.ones(10, dtype),
                    np.zeros(1, dtype),
                    np.ones(1, dtype),
                    np.empty(10, dtype)]

        for ary in ary_list:
            ary_gpu = gpuarray.to_gpu(ary)
            any_ary = np.any(ary)
            any_ary_gpu = ary_gpu.any().get()
            np.testing.assert_array_equal(any_ary_gpu, any_ary)
            assert any_ary_gpu.dtype == any_ary.dtype

        import itertools
        for _array in list(itertools.product([0, 1], [0, 1], [0, 1])):
            array = np.array(_array, dtype)
            array_gpu = gpuarray.to_gpu(array)
            any_array = np.any(array)
            any_array_gpu = array_gpu.any().get()

            np.testing.assert_array_equal(any_array_gpu, any_array)
            assert any_array_gpu.dtype == any_array.dtype

    @pytest.mark.parametrize("dtype", [np.int32, np.bool, np.float32, np.float64])
    def test_all(self, dtype):

        ary_list = [np.ones(10, dtype),
                    np.zeros(1, dtype),
                    np.ones(1, dtype),
                    np.empty(10, dtype)]

        for ary in ary_list:
            ary_gpu = gpuarray.to_gpu(ary)
            all_ary = np.all(ary)
            all_ary_gpu = ary_gpu.all().get()
            np.testing.assert_array_equal(all_ary_gpu, all_ary)
            assert all_ary_gpu.dtype == all_ary.dtype

        import itertools
        for _array in list(itertools.product([0, 1], [0, 1], [0, 1])):
            array = np.array(_array, dtype)
            array_gpu = gpuarray.to_gpu(array)
            all_array = np.all(array)
            all_array_gpu = array_gpu.all().get()

            np.testing.assert_array_equal(all_array_gpu, all_array)
            assert all_array_gpu.dtype == all_array.dtype

    def test_minmax(self):
        from pycuda.curandom import rand as curand

        if has_double_support():
            dtypes = [np.float64, np.float32, np.int32]
        else:
            dtypes = [np.float32, np.int32]

        for what in ["min", "max"]:
            for dtype in dtypes:
                a_gpu = curand((200000,), dtype)
                a = a_gpu.get()

                op_a = getattr(np, what)(a)
                op_a_gpu = getattr(gpuarray, what)(a_gpu).get()

                assert op_a_gpu == op_a, (op_a_gpu, op_a, dtype, what)

    def test_subset_minmax(self):
        from pycuda.curandom import rand as curand

        l_a = 200000
        gran = 5
        l_m = l_a - l_a // gran + 1

        if has_double_support():
            dtypes = [np.float64, np.float32, np.int32]
        else:
            dtypes = [np.float32, np.int32]

        for dtype in dtypes:
            a_gpu = curand((l_a,), dtype)
            a = a_gpu.get()

            meaningful_indices_gpu = gpuarray.zeros(l_m, dtype=np.int32)
            meaningful_indices = meaningful_indices_gpu.get()
            j = 0
            for i in range(len(meaningful_indices)):
                meaningful_indices[i] = j
                j = j + 1
                if j % gran == 0:
                    j = j + 1

            meaningful_indices_gpu = gpuarray.to_gpu(meaningful_indices)
            b = a[meaningful_indices]

            min_a = np.min(b)
            min_a_gpu = gpuarray.subset_min(meaningful_indices_gpu, a_gpu).get()

            assert min_a_gpu == min_a

    @pytest.mark.parametrize("sz", [2,
                                    3,
                                    4,
                                    5,
                                    6,
                                    7,
                                    31,
                                    32,
                                    33,
                                    127,
                                    128,
                                    129,
                                    255,
                                    256,
                                    257,
                                    16384 - 993,
                                    20000,
                                    ])
    def test_dot(self, sz):
        from pycuda.curandom import rand as curand

        a_gpu = curand((sz,))
        a = a_gpu.get()
        b_gpu = curand((sz,))
        b = b_gpu.get()

        dot_ab = np.dot(a, b)

        dot_ab_gpu = gpuarray.dot(a_gpu, b_gpu).get()

        assert abs(dot_ab_gpu - dot_ab) / abs(dot_ab) < 1e-4

    def test_slice(self):
        from pycuda.curandom import rand as curand

        sz = 20000
        a_gpu = curand((sz,))
        a = a_gpu.get()

        from random import randrange

        for i in range(200):
            start = randrange(sz)
            end = randrange(start, sz)

            a_gpu_slice = a_gpu[start:end]
            a_slice = a[start:end]

            assert la.norm(a_gpu_slice.get() - a_slice) == 0

    def test_2d_slice_c(self):
        from pycuda.curandom import rand as curand

        n = 1000
        m = 300
        a_gpu = curand((n, m))
        a = a_gpu.get()

        from random import randrange

        for i in range(200):
            start = randrange(n)
            end = randrange(start, n)

            a_gpu_slice = a_gpu[start:end]
            a_slice = a[start:end]

            assert la.norm(a_gpu_slice.get() - a_slice) == 0

    def test_2d_slice_f(self):
        from pycuda.curandom import rand as curand
        import pycuda.gpuarray as gpuarray

        n = 1000
        m = 300
        a_gpu = curand((n, m))
        a_gpu_f = gpuarray.GPUArray(
            (m, n), np.float32, gpudata=a_gpu.gpudata, order="F"
        )
        a = a_gpu_f.get()

        from random import randrange

        for i in range(200):
            start = randrange(n)
            end = randrange(start, n)

            a_gpu_slice = a_gpu_f[:, start:end]
            a_slice = a[:, start:end]

            assert la.norm(a_gpu_slice.get() - a_slice) == 0

    def test_where(self):
        a = np.array([1, 0, -1])
        b = np.array([2, 2, 2])
        c = np.array([3, 3, 3])

        import pycuda.gpuarray as gpuarray

        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = gpuarray.to_gpu(c)

        result = gpuarray.where(a_gpu, b_gpu, c_gpu).get()
        result_ref = np.where(a, b, c)

        np.testing.assert_allclose(result_ref, result, rtol=1e-5)

    def test_if_positive(self):
        from pycuda.curandom import rand as curand

        sz = 20
        a_gpu = curand((sz,))
        b_gpu = curand((sz,))
        a = a_gpu.get()
        b = b_gpu.get()

        import pycuda.gpuarray as gpuarray

        max_a_b_gpu = gpuarray.maximum(a_gpu, b_gpu)
        min_a_b_gpu = gpuarray.minimum(a_gpu, b_gpu)

        print(max_a_b_gpu)
        print(np.maximum(a, b))

        assert la.norm(max_a_b_gpu.get() - np.maximum(a, b)) == 0
        assert la.norm(min_a_b_gpu.get() - np.minimum(a, b)) == 0

    def test_take_put(self):
        for n in [5, 17, 333]:
            one_field_size = 8
            buf_gpu = gpuarray.zeros(n * one_field_size, dtype=np.float32)
            dest_indices = gpuarray.to_gpu(
                np.array([0, 1, 2, 3, 32, 33, 34, 35], dtype=np.uint32)
            )
            read_map = gpuarray.to_gpu(
                np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=np.uint32)
            )

            gpuarray.multi_take_put(
                arrays=[buf_gpu for i in range(n)],
                dest_indices=dest_indices,
                src_indices=read_map,
                src_offsets=[i * one_field_size for i in range(n)],
                dest_shape=(96,),
            )

            drv.Context.synchronize()

    def test_astype(self):
        from pycuda.curandom import rand as curand

        if not has_double_support():
            return

        a_gpu = curand((2000,), dtype=np.float32)

        a = a_gpu.get().astype(np.float64)
        a2 = a_gpu.astype(np.float64).get()

        assert a2.dtype == np.float64
        assert la.norm(a - a2) == 0, (a, a2)

        a_gpu = curand((2000,), dtype=np.float64)

        a = a_gpu.get().astype(np.float32)
        a2 = a_gpu.astype(np.float32).get()

        assert a2.dtype == np.float32
        assert la.norm(a - a2) / la.norm(a) < 1e-7

    def test_complex_bits(self):
        from pycuda.curandom import rand as curand

        if has_double_support():
            dtypes = [np.complex64, np.complex128]
        else:
            dtypes = [np.complex64]

        n = 20
        for tp in dtypes:
            dtype = np.dtype(tp)
            from pytools import match_precision

            real_dtype = match_precision(np.dtype(np.float64), dtype)

            z = curand((n,), real_dtype).astype(dtype) + 1j * curand(
                (n,), real_dtype
            ).astype(dtype)

            assert la.norm(z.get().real - z.real.get()) == 0
            assert la.norm(z.get().imag - z.imag.get()) == 0
            assert la.norm(z.get().conj() - z.conj().get()) == 0
            # verify conj with out parameter
            z_out = z.astype(np.complex64)
            assert z_out is z.conj(out=z_out)
            assert la.norm(z.get().conj() - z_out.get()) < 5e-6

            # verify contiguity is preserved
            for order in ["C", "F"]:
                # test both zero and non-zero value code paths
                z_real = gpuarray.zeros(z.shape, dtype=real_dtype, order=order)
                z2 = z.reshape(z.shape, order=order)
                for zdata in [z_real, z2]:
                    if order == "C":
                        assert zdata.flags.c_contiguous
                        assert zdata.real.flags.c_contiguous
                        assert zdata.imag.flags.c_contiguous
                        assert zdata.conj().flags.c_contiguous
                    elif order == "F":
                        assert zdata.flags.f_contiguous
                        assert zdata.real.flags.f_contiguous
                        assert zdata.imag.flags.f_contiguous
                        assert zdata.conj().flags.f_contiguous

    def test_pass_slice_to_kernel(self):
        mod = SourceModule(
            """
        __global__ void twice(float *a)
        {
          const int i = threadIdx.x + blockIdx.x * blockDim.x;
          a[i] *= 2;
        }
        """
        )

        multiply_them = mod.get_function("twice")

        a = np.ones(256 ** 2, np.float32)
        a_gpu = gpuarray.to_gpu(a)

        multiply_them(a_gpu[256:-256], block=(256, 1, 1), grid=(254, 1))

        a = a_gpu.get()
        assert (a[255:257] == np.array([1, 2], np.float32)).all()
        np.testing.assert_array_equal(a[255 * 256 - 1: 255 * 256 + 1],
                                      np.array([2, 1], np.float32))

    def test_scan(self):
        from pycuda.scan import ExclusiveScanKernel, InclusiveScanKernel

        for cls in [ExclusiveScanKernel, InclusiveScanKernel]:
            scan_kern = cls(np.int32, "a+b", "0")

            for n in [
                10,
                2 ** 10 - 5,
                2 ** 10,
                2 ** 20 - 2 ** 18,
                2 ** 20 - 2 ** 18 + 5,
                2 ** 10 + 5,
                2 ** 20 + 5,
                2 ** 20,
                2 ** 24,
            ]:
                host_data = np.random.randint(0, 10, n).astype(np.int32)
                gpu_data = gpuarray.to_gpu(host_data)

                scan_kern(gpu_data)

                desired_result = np.cumsum(host_data, axis=0)
                if cls is ExclusiveScanKernel:
                    desired_result -= host_data

                assert (gpu_data.get() == desired_result).all()

    def test_stride_preservation(self):
        A = np.random.rand(3, 3)
        AT = A.T
        print((AT.flags.f_contiguous, AT.flags.c_contiguous))
        AT_GPU = gpuarray.to_gpu(AT)
        print((AT_GPU.flags.f_contiguous, AT_GPU.flags.c_contiguous))
        assert np.allclose(AT_GPU.get(), AT)

    def test_vector_fill(self):
        a_gpu = gpuarray.GPUArray(100, dtype=gpuarray.vec.float3)
        a_gpu.fill(gpuarray.vec.make_float3(0.0, 0.0, 0.0))
        a = a_gpu.get()
        assert a.dtype == gpuarray.vec.float3

    def test_create_complex_zeros(self):
        gpuarray.zeros(3, np.complex64)

    def test_reshape(self):
        a = np.arange(128).reshape(8, 16).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)

        # different ways to specify the shape
        a_gpu.reshape(4, 32)
        a_gpu.reshape((4, 32))
        a_gpu.reshape([4, 32])

        # using -1 as unknown dimension
        assert a_gpu.reshape(-1, 32).shape == (4, 32)
        assert a_gpu.reshape((32, -1)).shape == (32, 4)
        assert a_gpu.reshape((8, -1, 4)).shape == (8, 4, 4)

        throws_exception = False
        try:
            a_gpu.reshape(-1, -1, 4)
        except ValueError:
            throws_exception = True
        assert throws_exception

        # with order specified
        a_gpu = a_gpu.reshape((4, 32), order="C")
        assert a_gpu.flags.c_contiguous
        a_gpu = a_gpu.reshape(4, 32, order="F")
        assert a_gpu.flags.f_contiguous
        a_gpu = a_gpu.reshape((4, 32), order="F")
        assert a_gpu.flags.f_contiguous
        # default is C-contiguous
        a_gpu = a_gpu.reshape((4, 32))
        assert a_gpu.flags.c_contiguous

    def test_view(self):
        a = np.arange(128).reshape(8, 16).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)

        # same dtype
        view = a_gpu.view()
        assert view.shape == a_gpu.shape and view.dtype == a_gpu.dtype

        # larger dtype
        view = a_gpu.view(np.complex64)
        assert view.shape == (8, 8) and view.dtype == np.complex64

        # smaller dtype
        view = a_gpu.view(np.int16)
        assert view.shape == (8, 32) and view.dtype == np.int16

    def test_squeeze(self):
        shape = (40, 2, 5, 100)
        a_cpu = np.random.random(size=shape)
        a_gpu = gpuarray.to_gpu(a_cpu)

        # Slice with length 1 on dimensions 0 and 1
        a_gpu_slice = a_gpu[0:1, 1:2, :, :]
        assert a_gpu_slice.shape == (1, 1, shape[2], shape[3])
        assert a_gpu_slice.flags.c_contiguous

        # Squeeze it and obtain contiguity
        a_gpu_squeezed_slice = a_gpu[0:1, 1:2, :, :].squeeze()
        assert a_gpu_squeezed_slice.shape == (shape[2], shape[3])
        assert a_gpu_squeezed_slice.flags.c_contiguous

        # Check that we get the original values out
        np.testing.assert_array_equal(a_gpu_slice.get().ravel(),
                                      a_gpu_squeezed_slice.get().ravel())

        # Slice with length 1 on dimensions 2
        a_gpu_slice = a_gpu[:, :, 2:3, :]
        assert a_gpu_slice.shape == (shape[0], shape[1], 1, shape[3])
        assert not a_gpu_slice.flags.c_contiguous

        # Squeeze it, but no contiguity here
        a_gpu_squeezed_slice = a_gpu[:, :, 2:3, :].squeeze()
        assert a_gpu_squeezed_slice.shape == (shape[0], shape[1], shape[3])
        assert not a_gpu_squeezed_slice.flags.c_contiguous

        # Check that we get the original values out
        np.testing.assert_array_equal(a_gpu_slice.get().ravel(),
                                      a_gpu_squeezed_slice.get().ravel())

    def test_struct_reduce(self):
        preamble = """
        struct minmax_collector
        {
            float cur_min;
            float cur_max;

            __device__
            minmax_collector()
            { }

            __device__
            minmax_collector(float cmin, float cmax)
            : cur_min(cmin), cur_max(cmax)
            { }

            __device__ minmax_collector(minmax_collector const &src)
            : cur_min(src.cur_min), cur_max(src.cur_max)
            { }

            __device__ minmax_collector(minmax_collector const volatile &src)
            : cur_min(src.cur_min), cur_max(src.cur_max)
            { }

            __device__ minmax_collector volatile &operator=(
                minmax_collector const &src) volatile
            {
                cur_min = src.cur_min;
                cur_max = src.cur_max;
                return *this;
            }
        };

        __device__
        minmax_collector agg_mmc(minmax_collector a, minmax_collector b)
        {
            return minmax_collector(
                fminf(a.cur_min, b.cur_min),
                fmaxf(a.cur_max, b.cur_max));
        }
        """
        mmc_dtype = np.dtype([("cur_min", np.float32), ("cur_max", np.float32)])

        from pycuda.curandom import rand as curand

        a_gpu = curand((20000,), dtype=np.float32)
        a = a_gpu.get()

        from pycuda.tools import register_dtype

        register_dtype(mmc_dtype, "minmax_collector")

        from pycuda.reduction import ReductionKernel

        red = ReductionKernel(
            mmc_dtype,
            neutral="minmax_collector(10000, -10000)",
            # FIXME: needs infinity literal in real use, ok here
            reduce_expr="agg_mmc(a, b)",
            map_expr="minmax_collector(x[i], x[i])",
            arguments="float *x",
            preamble=preamble,
        )

        minmax = red(a_gpu).get()
        # print minmax["cur_min"], minmax["cur_max"]
        # print np.min(a), np.max(a)

        assert minmax["cur_min"] == np.min(a)
        assert minmax["cur_max"] == np.max(a)

    def test_reduce_out(self):
        from pycuda.curandom import rand as curand

        a_gpu = curand((10, 200), dtype=np.float32)
        a = a_gpu.get()

        from pycuda.reduction import ReductionKernel

        red = ReductionKernel(
            np.float32, neutral=0, reduce_expr="max(a,b)", arguments="float *in"
        )
        max_gpu = gpuarray.empty(10, dtype=np.float32)
        for i in range(10):
            red(a_gpu[i], out=max_gpu[i])

        assert np.alltrue(a.max(axis=1) == max_gpu.get())

    def test_sum_allocator(self):
        # FIXME
        from pytest import skip

        skip("https://github.com/inducer/pycuda/issues/163")
        # crashes with  terminate called after throwing an instance
        # of 'pycuda::error'
        # what():  explicit_context_dependent failed: invalid device context -
        # no currently active context?

        import pycuda.tools

        pool = pycuda.tools.DeviceMemoryPool()

        rng = np.random.randint(low=512, high=1024)

        a = gpuarray.arange(rng, dtype=np.int32)
        b = gpuarray.sum(a)
        c = gpuarray.sum(a, allocator=pool.allocate)

        # Test that we get the correct results
        assert b.get() == rng * (rng - 1) // 2
        assert c.get() == rng * (rng - 1) // 2

        # Test that result arrays were allocated with the appropriate allocator
        assert b.allocator == a.allocator
        assert c.allocator == pool.allocate

    def test_dot_allocator(self):
        # FIXME
        from pytest import skip

        skip("https://github.com/inducer/pycuda/issues/163")

        import pycuda.tools

        pool = pycuda.tools.DeviceMemoryPool()

        a_cpu = np.random.randint(low=512, high=1024, size=1024)
        b_cpu = np.random.randint(low=512, high=1024, size=1024)

        # Compute the result on the CPU
        dot_cpu_1 = np.dot(a_cpu, b_cpu)

        a_gpu = gpuarray.to_gpu(a_cpu)
        b_gpu = gpuarray.to_gpu(b_cpu)

        # Compute the result on the GPU using different allocators
        dot_gpu_1 = gpuarray.dot(a_gpu, b_gpu)
        dot_gpu_2 = gpuarray.dot(a_gpu, b_gpu, allocator=pool.allocate)

        # Test that we get the correct results
        assert dot_cpu_1 == dot_gpu_1.get()
        assert dot_cpu_1 == dot_gpu_2.get()

        # Test that result arrays were allocated with the appropriate allocator
        assert dot_gpu_1.allocator == a_gpu.allocator
        assert dot_gpu_2.allocator == pool.allocate

    def test_view_and_strides(self):
        from pycuda.curandom import rand as curand

        X = curand((5, 10), dtype=np.float32)
        Y = X[:3, :5]
        y = Y.view()

        assert y.shape == Y.shape
        assert y.strides == Y.strides

        assert np.array_equal(y.get(), X.get()[:3, :5])

    def test_scalar_comparisons(self):
        a = np.array([1.0, 0.25, 0.1, -0.1, 0.0])
        a_gpu = gpuarray.to_gpu(a)

        x_gpu = a_gpu > 0.25
        x = (a > 0.25).astype(a.dtype)
        assert (x == x_gpu.get()).all()

        x_gpu = a_gpu <= 0.25
        x = (a <= 0.25).astype(a.dtype)
        assert (x == x_gpu.get()).all()

        x_gpu = a_gpu == 0.25
        x = (a == 0.25).astype(a.dtype)
        assert (x == x_gpu.get()).all()

        x_gpu = a_gpu == 1  # using an integer scalar
        x = (a == 1).astype(a.dtype)
        assert (x == x_gpu.get()).all()

    def test_minimum_maximum_scalar(self):
        from pycuda.curandom import rand as curand

        sz = 20
        a_gpu = curand((sz,))
        a = a_gpu.get()

        import pycuda.gpuarray as gpuarray

        max_a0_gpu = gpuarray.maximum(a_gpu, 0)
        min_a0_gpu = gpuarray.minimum(0, a_gpu)

        assert la.norm(max_a0_gpu.get() - np.maximum(a, 0)) == 0
        assert la.norm(min_a0_gpu.get() - np.minimum(0, a)) == 0

    def test_transpose(self):
        from pycuda.curandom import rand as curand

        a_gpu = curand((10, 20, 30))
        a = a_gpu.get()

        assert np.allclose(a_gpu.T.get(), a.T)

    def test_newaxis(self):
        from pycuda.curandom import rand as curand

        a_gpu = curand((10, 20, 30))
        a = a_gpu.get()

        b_gpu = a_gpu[:, np.newaxis]
        b = a[:, np.newaxis]

        assert b_gpu.shape == b.shape
        assert b_gpu.strides == b.strides

    def test_copy(self):
        from pycuda.curandom import rand as curand

        a_gpu = curand((3, 3))

        for start, stop, step in [(0, 3, 1), (1, 2, 1), (0, 3, 2), (0, 3, 3)]:
            assert np.allclose(
                a_gpu[start:stop:step].get(), a_gpu.get()[start:stop:step]
            )

        a_gpu = curand((3, 1))
        for start, stop, step in [(0, 3, 1), (1, 2, 1), (0, 3, 2), (0, 3, 3)]:
            assert np.allclose(
                a_gpu[start:stop:step].get(), a_gpu.get()[start:stop:step]
            )

        a_gpu = curand((3, 3, 3))
        for start, stop, step in [(0, 3, 1), (1, 2, 1), (0, 3, 2), (0, 3, 3)]:
            assert np.allclose(
                a_gpu[start:stop:step, start:stop:step].get(),
                a_gpu.get()[start:stop:step, start:stop:step],
            )

        a_gpu = curand((3, 3, 3)).transpose((1, 2, 0))
        for start, stop, step in [(0, 3, 1), (1, 2, 1), (0, 3, 2), (0, 3, 3)]:
            assert np.allclose(
                a_gpu[start:stop:step, :, start:stop:step].get(),
                a_gpu.get()[start:stop:step, :, start:stop:step],
            )

        # 4-d should work as long as only 2 axes are discontiguous
        a_gpu = curand((3, 3, 3, 3))
        for start, stop, step in [(0, 3, 1), (1, 2, 1), (0, 3, 3)]:
            assert np.allclose(
                a_gpu[start:stop:step, :, start:stop:step].get(),
                a_gpu.get()[start:stop:step, :, start:stop:step],
            )

    def test_get_set(self):
        import pycuda.gpuarray as gpuarray

        a = np.random.normal(0.0, 1.0, (4, 4))
        a_gpu = gpuarray.to_gpu(a)
        assert np.allclose(a_gpu.get(), a)
        assert np.allclose(a_gpu[1:3, 1:3].get(), a[1:3, 1:3])

        a = np.random.normal(0.0, 1.0, (4, 4, 4)).transpose((1, 2, 0))
        a_gpu = gpuarray.to_gpu(a)
        assert np.allclose(a_gpu.get(), a)
        assert np.allclose(a_gpu[1:3, 1:3, 1:3].get(), a[1:3, 1:3, 1:3])

    def test_zeros_like_etc(self):
        shape = (16, 16)
        a = np.random.randn(*shape).astype(np.float32)
        z = gpuarray.to_gpu(a)
        zf = gpuarray.to_gpu(np.asfortranarray(a))
        a_noncontig = np.arange(3 * 4 * 5).reshape(3, 4, 5).swapaxes(1, 2)
        z_noncontig = gpuarray.to_gpu(a_noncontig)
        for func in [gpuarray.empty_like, gpuarray.zeros_like, gpuarray.ones_like]:
            for arr in [z, zf, z_noncontig]:

                contig = arr.flags.c_contiguous or arr.flags.f_contiguous

                if not contig:
                    continue

                # Output matches order of input.
                # Non-contiguous becomes C-contiguous
                new_z = func(arr, order="A")
                if contig:
                    assert new_z.flags.c_contiguous == arr.flags.c_contiguous
                    assert new_z.flags.f_contiguous == arr.flags.f_contiguous
                else:
                    assert new_z.flags.c_contiguous is True
                    assert new_z.flags.f_contiguous is False
                assert new_z.dtype == arr.dtype
                assert new_z.shape == arr.shape

                # Force C-ordered output
                new_z = func(arr, order="C")
                assert new_z.flags.c_contiguous is True
                assert new_z.flags.f_contiguous is False
                assert new_z.dtype == arr.dtype
                assert new_z.shape == arr.shape

                # Force Fortran-orded output
                new_z = func(arr, order="F")
                assert new_z.flags.c_contiguous is False
                assert new_z.flags.f_contiguous is True
                assert new_z.dtype == arr.dtype
                assert new_z.shape == arr.shape

                # Change the dtype, but otherwise match order & strides
                # order = "K" so non-contiguous array remains non-contiguous
                new_z = func(arr, dtype=np.complex64, order="K")
                assert new_z.flags.c_contiguous == arr.flags.c_contiguous
                assert new_z.flags.f_contiguous == arr.flags.f_contiguous
                assert new_z.dtype == np.complex64
                assert new_z.shape == arr.shape

    def test_logical_and_or(self):
        rng = np.random.default_rng(seed=0)
        for op in ["logical_and", "logical_or"]:
            x_np = rng.random((10, 4))
            y_np = rng.random((10, 4))
            zeros_np = np.zeros((10, 4))
            ones_np = np.ones((10, 4))

            x_cu = gpuarray.to_gpu(x_np)
            y_cu = gpuarray.to_gpu(y_np)
            zeros_cu = gpuarray.zeros((10, 4), "float64")
            ones_cu = gpuarray.ones((10, 4))

            np.testing.assert_array_equal(
                getattr(gpuarray, op)(x_cu, y_cu).get(),
                getattr(np, op)(x_np, y_np))
            np.testing.assert_array_equal(
                getattr(gpuarray, op)(x_cu, ones_cu).get(),
                getattr(np, op)(x_np, ones_np))
            np.testing.assert_array_equal(
                getattr(gpuarray, op)(x_cu, zeros_cu).get(),
                getattr(np, op)(x_np, zeros_np))
            np.testing.assert_array_equal(
                getattr(gpuarray, op)(x_cu, 1.0).get(),
                getattr(np, op)(x_np, ones_np))
            np.testing.assert_array_equal(
                getattr(gpuarray, op)(x_cu, 0.0).get(),
                getattr(np, op)(x_np, 0.0))

    def test_logical_not(self):
        rng = np.random.default_rng(seed=0)
        x_np = rng.random((10, 4))
        x_cu = gpuarray.to_gpu(x_np)

        np.testing.assert_array_equal(
            gpuarray.logical_not(x_cu).get(),
            np.logical_not(x_np))
        np.testing.assert_array_equal(
            gpuarray.logical_not(gpuarray.zeros(10, "float64")).get(),
            np.logical_not(np.zeros(10)))
        np.testing.assert_array_equal(
            gpuarray.logical_not(gpuarray.ones(10)).get(),
            np.logical_not(np.ones(10)))

    def test_truth_value(self):
        for i in range(5):
            shape = (1,)*i
            zeros = gpuarray.zeros(shape, dtype="float32")
            ones = gpuarray.ones(shape, dtype="float32")
            assert bool(ones)
            assert not bool(zeros)

    def test_setitem_scalar(self):
        a = gpuarray.zeros(5, "float64") + 42
        np.testing.assert_allclose(a.get(), 42)
        a[...] = 1729
        np.testing.assert_allclose(a.get(), 1729)

    def test_default_zero(self):
        # This test was added to make sure that
        # gpurray.zeros is using np.float64 as the default dtype arg
        a_gpu = gpuarray.zeros(10)
        assert a_gpu.dtype == np.float64

    @pytest.mark.parametrize("dtype,rtol", [(np.complex64, 1e-6),
                                            (np.complex128, 1e-14)])
    def test_log10(self, dtype, rtol):
        from pycuda import cumath

        rng = np.random.default_rng(seed=0)
        x_np = rng.random((10, 4)) + dtype(1j)*rng.random((10, 4))
        x_cu = gpuarray.to_gpu(x_np)
        np.testing.assert_allclose(cumath.log10(x_cu).get(), np.log10(x_np),
                                   rtol=rtol)

    @pytest.mark.parametrize("ldtype", [np.int32, np.int64,
                                        np.float32, np.float64,
                                        np.complex64, np.complex128])
    @pytest.mark.parametrize("rdtype", [np.int32, np.int64,
                                        np.float32, np.float64,
                                        np.complex64, np.complex128])
    @pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul,
                                    operator.truediv])
    def test_binary_ops_with_unequal_dtypes(self, ldtype, rdtype, op):
        # See https://github.com/inducer/pycuda/issues/372
        if op == operator.truediv and {ldtype, rdtype} <= {np.int32, np.int64}:
            pytest.xfail("Enable after"
                         " gitlab.tiker.net/inducer/pycuda/-/merge_requests/66"
                         "is merged.")

        rng = np.random.default_rng(0)
        lop_np = get_random_array(rng, (10, 4), ldtype)
        rop_np = get_random_array(rng, (10, 4), rdtype)

        expected_result = op(lop_np, rop_np)
        result = op(gpuarray.to_gpu(lop_np), gpuarray.to_gpu(rop_np)).get()

        assert result.dtype == expected_result.dtype
        assert result.shape == expected_result.shape
        np.testing.assert_allclose(expected_result, result,
                                   rtol=5e-5)

    def test_big_array_elementwise(self):
        skip_if_not_enough_gpu_memory(4.5)

        from pycuda.elementwise import ElementwiseKernel
        n_items = 2**32

        eltwise = ElementwiseKernel(
            "unsigned char* d_arr",
            "d_arr[i] = (unsigned char) (i & 0b11111111)", "mod_linspace"
        )
        d_arr = gpuarray.empty(n_items, np.uint8)
        eltwise(d_arr)
        result = d_arr.get()[()]
        # Needs 8.6 GB memory on host - numpy cannot keep uint8 for mod() operation,
        # and np.mod(np.arange()) is way too slow
        reference = np.mod(np.arange(d_arr.size, dtype=np.int16), 256, dtype=np.int16)
        reference -= result
        assert np.max(reference) == 0

    def test_big_array_reduction(self):
        skip_if_not_enough_gpu_memory(4.5)

        from pycuda.reduction import ReductionKernel
        n_items = 2**32 + 11
        reduction = ReductionKernel(
            np.uint8,
            neutral="0",
            reduce_expr="(a+b) & 0b11111111",
            map_expr="x[i]",
            arguments="unsigned char* x"
        )
        d_arr = gpuarray.zeros(n_items, np.uint8)
        d_arr.fill(1)  # elementwise!
        result = reduction(d_arr).get()[()]

        assert result == 11

    def test_big_array_scan(self):
        skip_if_not_enough_gpu_memory(4.5)
        n_items = 2**32 + 12
        from pycuda.scan import InclusiveScanKernel

        cumsum = InclusiveScanKernel(np.uint8, "(a+b) & 0b11111111")
        d_arr = gpuarray.zeros(n_items, np.uint8)
        d_arr.fill(1)
        result = cumsum(d_arr).get()[()]
        # Needs 8.6 GB on host. numpy.allclose() is way too slow otherwise.
        reference = np.tile(
            np.roll(np.arange(256, dtype=np.int16), -1), n_items//256
        )
        reference -= result[:reference.size]
        assert np.max(reference) == 0
        assert np.allclose(result[2**32:], np.arange(1, 12+1))

    def test_noncontig_transpose(self):
        # https://github.com/inducer/pycuda/issues/385
        d = gpuarray.zeros((1000, 15, 2048), "f")
        d.transpose(axes=(1, 0, 2))  # works
        d2 = d[:, 7:9, :]  # non C-contiguous
        d2.transpose(axes=(1, 0, 2))  # crashes for recent versions


if __name__ == "__main__":
    # make sure that import failures get reported, instead of skipping the tests.
    import pycuda.autoinit  # noqa

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main

        main([__file__])
