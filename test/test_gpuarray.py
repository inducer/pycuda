#! /usr/bin/env python
import numpy as np
import numpy.linalg as la
import sys
from pycuda.tools import mark_cuda_test
from pycuda.characterize import has_double_support




def have_pycuda():
    try:
        import pycuda
        return True
    except:
        return False

if have_pycuda():
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule




class TestGPUArray:
    disabled = not have_pycuda()

    @mark_cuda_test
    def test_pow_array(self):
        a = np.array([1,2,3,4,5]).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)

        result = pow(a_gpu,a_gpu).get()
        assert (np.abs(a**a - result) < 1e-3).all()

        result = (a_gpu**a_gpu).get()
        assert (np.abs(pow(a, a) - result) < 1e-3).all()




    @mark_cuda_test
    def test_pow_number(self):
        a = np.array([1,2,3,4,5,6,7,8,9,10]).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)

        result = pow(a_gpu, 2).get()
        assert (np.abs(a**2 - result) < 1e-3).all()



    @mark_cuda_test
    def test_abs(self):
        a = -gpuarray.arange(111, dtype=np.float32)
        res = a.get()

        for i in range(111):
            assert res[i] <= 0

        a = abs(a)

        res = a.get()

        for i in range (111):
            assert abs(res[i]) >= 0
            assert res[i] == i


    @mark_cuda_test
    def test_len(self):
        a = np.array([1,2,3,4,5,6,7,8,9,10]).astype(np.float32)
        a_cpu = gpuarray.to_gpu(a)
        assert len(a_cpu) == 10




    @mark_cuda_test
    def test_multiply(self):
        """Test the muliplication of an array with a scalar. """

        for sz in [10, 50000]:
            for dtype, scalars in [
                (np.float32, [2]),
                (np.complex64, [2, 2j])
                ]:
                for scalar in scalars:
                    a = np.arange(sz).astype(dtype)
                    a_gpu = gpuarray.to_gpu(a)
                    a_doubled = (scalar * a_gpu).get()

                    assert (a * scalar == a_doubled).all()

    @mark_cuda_test
    def test_multiply_array(self):
        """Test the multiplication of two arrays."""

        a = np.array([1,2,3,4,5,6,7,8,9,10]).astype(np.float32)

        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(a)

        a_squared = (b_gpu*a_gpu).get()

        assert (a*a == a_squared).all()




    @mark_cuda_test
    def test_addition_array(self):
        """Test the addition of two arrays."""

        a = np.array([1,2,3,4,5,6,7,8,9,10]).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)
        a_added = (a_gpu+a_gpu).get()

        assert (a+a == a_added).all()




    @mark_cuda_test
    def test_iaddition_array(self):
        """Test the inplace addition of two arrays."""

        a = np.array([1,2,3,4,5,6,7,8,9,10]).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)
        a_gpu += a_gpu
        a_added = a_gpu.get()

        assert (a+a == a_added).all()



    @mark_cuda_test
    def test_addition_scalar(self):
        """Test the addition of an array and a scalar."""

        a = np.array([1,2,3,4,5,6,7,8,9,10]).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)
        a_added = (7+a_gpu).get()

        assert (7+a == a_added).all()



    @mark_cuda_test
    def test_iaddition_scalar(self):
        """Test the inplace addition of an array and a scalar."""

        a = np.array([1,2,3,4,5,6,7,8,9,10]).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)
        a_gpu += 7
        a_added = a_gpu.get()

        assert (7+a == a_added).all()




    @mark_cuda_test
    def test_substract_array(self):
        """Test the substraction of two arrays."""
        #test data
        a = np.array([1,2,3,4,5,6,7,8,9,10]).astype(np.float32)
        b = np.array([10,20,30,40,50,60,70,80,90,100]).astype(np.float32)

        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)

        result = (a_gpu-b_gpu).get()
        assert (a-b == result).all()

        result = (b_gpu-a_gpu).get()
        assert (b-a == result).all()




    @mark_cuda_test
    def test_substract_scalar(self):
        """Test the substraction of an array and a scalar."""

        #test data
        a = np.array([1,2,3,4,5,6,7,8,9,10]).astype(np.float32)

        #convert a to a gpu object
        a_gpu = gpuarray.to_gpu(a)

        result = (a_gpu-7).get()
        assert (a-7 == result).all()

        result = (7-a_gpu).get()
        assert (7-a == result).all()




    @mark_cuda_test
    def test_divide_scalar(self):
        """Test the division of an array and a scalar."""

        a = np.array([1,2,3,4,5,6,7,8,9,10]).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)

        result = (a_gpu/2).get()
        assert (a/2 == result).all()

        result = (2/a_gpu).get()
        assert (2/a == result).all()




    @mark_cuda_test
    def test_divide_array(self):
        """Test the division of an array and a scalar. """

        #test data
        a = np.array([10,20,30,40,50,60,70,80,90,100]).astype(np.float32)
        b = np.array([10,10,10,10,10,10,10,10,10,10]).astype(np.float32)

        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)

        a_divide = (a_gpu/b_gpu).get()
        assert (np.abs(a/b - a_divide) < 1e-3).all()

        a_divide = (b_gpu/a_gpu).get()
        assert (np.abs(b/a - a_divide) < 1e-3).all()




    @mark_cuda_test
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




    @mark_cuda_test
    def test_curand_wrappers(self):
        from pycuda.curandom import get_curand_version
        if get_curand_version() is None:
            from pytest import skip
            skip("curand not installed")


        from pycuda.curandom import (
                XORWOWRandomNumberGenerator,
                Sobol32RandomNumberGenerator)

        if has_double_support():
            dtypes = [np.float32, np.float64]
        else:
            dtypes = [np.float32]

        for gen_type in [
                XORWOWRandomNumberGenerator,
                #Sobol32RandomNumberGenerator
                ]:
            gen = gen_type()

            for dtype in dtypes:
                gen.gen_normal(10000, dtype)
                # test non-Box-Muller version, if available
                gen.gen_normal(10001, dtype)

                x = gen.gen_uniform(10000, dtype)
                x_host = x.get()
                assert (-1 <= x_host).all()
                assert (x_host <= 1).all()

            gen.gen_uniform(10000, np.uint32)



    @mark_cuda_test
    def test_array_gt(self):
        """Test whether array contents are > the other array's
        contents"""

        a = np.array([5,10]).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)
        b = np.array([2,10]).astype(np.float32)
        b_gpu = gpuarray.to_gpu(b)
        result = (a_gpu > b_gpu).get()
        assert result[0] == True
        assert result[1] == False

    @mark_cuda_test
    def test_array_lt(self):
        """Test whether array contents are < the other array's
        contents"""

        a = np.array([5,10]).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)
        b = np.array([2,10]).astype(np.float32)
        b_gpu = gpuarray.to_gpu(b)
        result = (b_gpu < a_gpu).get()
        assert result[0] == True
        assert result[1] == False

    @mark_cuda_test
    def test_array_le(self):
        """Test whether array contents are <= the other array's
        contents"""

        a = np.array([5,10, 1]).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)
        b = np.array([2,10, 2]).astype(np.float32)
        b_gpu = gpuarray.to_gpu(b)
        result = (b_gpu <= a_gpu).get()
        assert result[0] == True
        assert result[1] == True
        assert result[2] == False

    @mark_cuda_test
    def test_array_ge(self):
        """Test whether array contents are >= the other array's
        contents"""

        a = np.array([5,10,1]).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)
        b = np.array([2,10,2]).astype(np.float32)
        b_gpu = gpuarray.to_gpu(b)
        result = (a_gpu >= b_gpu).get()
        assert result[0] == True
        assert result[1] == True
        assert result[2] == False

    @mark_cuda_test
    def test_array_eq(self):
        """Test whether array contents are == the other array's
        contents"""

        a = np.array([5,10]).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)
        b = np.array([2,10]).astype(np.float32)
        b_gpu = gpuarray.to_gpu(b)
        result = (a_gpu == b_gpu).get()
        assert result[0] == False
        assert result[1] == True

    @mark_cuda_test
    def test_array_ne(self):
        """Test whether array contents are != the other array's
        contents"""

        a = np.array([5,10]).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)
        b = np.array([2,10]).astype(np.float32)
        b_gpu = gpuarray.to_gpu(b)
        result = (a_gpu != b_gpu).get()
        assert result[0] == True
        assert result[1] == False


    @mark_cuda_test
    def test_nan_arithmetic(self):
        def make_nan_contaminated_vector(size):
            shape = (size,)
            a = np.random.randn(*shape).astype(np.float32)
            #for i in range(0, shape[0], 3):
                #a[i] = float('nan')
            from random import randrange
            for i in range(size//10):
                a[randrange(0, size)] = float('nan')
            return a

        size = 1 << 20

        a = make_nan_contaminated_vector(size)
        a_gpu = gpuarray.to_gpu(a)
        b = make_nan_contaminated_vector(size)
        b_gpu = gpuarray.to_gpu(b)

        ab = a*b
        ab_gpu = (a_gpu*b_gpu).get()

        assert (np.isnan(ab) == np.isnan(ab_gpu)).all()




    @mark_cuda_test
    def test_elwise_kernel(self):
        from pycuda.curandom import rand as curand

        a_gpu = curand((50,))
        b_gpu = curand((50,))

        from pycuda.elementwise import ElementwiseKernel
        lin_comb = ElementwiseKernel(
                "float a, float *x, float b, float *y, float *z",
                "z[i] = a*x[i] + b*y[i]",
                "linear_combination")

        c_gpu = gpuarray.empty_like(a_gpu)
        lin_comb(5, a_gpu, 6, b_gpu, c_gpu)

        assert la.norm((c_gpu - (5*a_gpu+6*b_gpu)).get()) < 1e-5




    @mark_cuda_test
    def test_ranged_elwise_kernel(self):
        from pycuda.elementwise import ElementwiseKernel
        set_to_seven = ElementwiseKernel(
                "float *z",
                "z[i] = 7",
                "set_to_seven")

        for i, slc in enumerate([
                slice(5, 20000),
                slice(5, 20000, 17),
                slice(3000, 5, -1),
                slice(1000, -1),
                ]):

            a_gpu = gpuarray.zeros((50000,), dtype=np.float32)
            a_cpu = np.zeros(a_gpu.shape, a_gpu.dtype)

            a_cpu[slc] = 7
            set_to_seven(a_gpu, slice=slc)
            drv.Context.synchronize()

            assert la.norm(a_cpu - a_gpu.get()) == 0, i




    @mark_cuda_test
    def test_take(self):
        idx = gpuarray.arange(0, 200000, 2, dtype=np.uint32)
        a = gpuarray.arange(0, 600000, 3, dtype=np.float32)
        result = gpuarray.take(a, idx)
        assert ((3*idx).get() == result.get()).all()




    @mark_cuda_test
    def test_arange(self):
        a = gpuarray.arange(12, dtype=np.float32)
        assert (np.arange(12, dtype=np.float32) == a.get()).all()




    @mark_cuda_test
    def test_reverse(self):
        a = np.array([1,2,3,4,5,6,7,8,9,10]).astype(np.float32)
        a_cpu = gpuarray.to_gpu(a)

        a_cpu = a_cpu.reverse()


        b = a_cpu.get()

        for i in range(0,10):
            assert a[len(a)-1-i] == b[i]

    @mark_cuda_test
    def test_sum(self):
        from pycuda.curandom import rand as curand
        a_gpu = curand((200000,))
        a = a_gpu.get()

        sum_a = np.sum(a)

        from pycuda.reduction import get_sum_kernel
        sum_a_gpu = gpuarray.sum(a_gpu).get()

        assert abs(sum_a_gpu-sum_a)/abs(sum_a) < 1e-4

    @mark_cuda_test
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

    @mark_cuda_test
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

    @mark_cuda_test
    def test_dot(self):
        from pycuda.curandom import rand as curand
        a_gpu = curand((200000,))
        a = a_gpu.get()
        b_gpu = curand((200000,))
        b = b_gpu.get()

        dot_ab = np.dot(a, b)

        dot_ab_gpu = gpuarray.dot(a_gpu, b_gpu).get()

        assert abs(dot_ab_gpu-dot_ab)/abs(dot_ab) < 1e-4

    @mark_cuda_test
    def test_slice(self):
        from pycuda.curandom import rand as curand

        l = 20000
        a_gpu = curand((l,))
        a = a_gpu.get()

        from random import randrange
        for i in range(200):
            start = randrange(l)
            end = randrange(start, l)

            a_gpu_slice = a_gpu[start:end]
            a_slice = a[start:end]

            assert la.norm(a_gpu_slice.get()-a_slice) == 0

    @mark_cuda_test
    def test_if_positive(self):
        from pycuda.curandom import rand as curand

        l = 20
        a_gpu = curand((l,))
        b_gpu = curand((l,))
        a = a_gpu.get()
        b = b_gpu.get()

        import pycuda.gpuarray as gpuarray

        max_a_b_gpu = gpuarray.maximum(a_gpu, b_gpu)
        min_a_b_gpu = gpuarray.minimum(a_gpu, b_gpu)

        print max_a_b_gpu
        print np.maximum(a, b)

        assert la.norm(max_a_b_gpu.get()- np.maximum(a, b)) == 0
        assert la.norm(min_a_b_gpu.get()- np.minimum(a, b)) == 0

    @mark_cuda_test
    def test_take_put(self):
        for n in [5, 17, 333]:
            one_field_size = 8
            buf_gpu = gpuarray.zeros(n*one_field_size, dtype=np.float32)
            dest_indices = gpuarray.to_gpu(np.array([ 0,  1,  2,  3, 32, 33, 34, 35], dtype=np.uint32))
            read_map = gpuarray.to_gpu(np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=np.uint32))

            gpuarray.multi_take_put(
                    arrays=[buf_gpu for i in range(n)],
                    dest_indices=dest_indices,
                    src_indices=read_map,
                    src_offsets=[i*one_field_size for i in range(n)],
                    dest_shape=(96,))

            drv.Context.synchronize()

    @mark_cuda_test
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
        assert la.norm(a - a2)/la.norm(a) < 1e-7

    @mark_cuda_test
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

            z = (curand((n,), real_dtype).astype(dtype)
                    + 1j*curand((n,), real_dtype).astype(dtype))

            assert la.norm(z.get().real - z.real.get()) == 0
            assert la.norm(z.get().imag - z.imag.get()) == 0
            assert la.norm(z.get().conj() - z.conj().get()) == 0

    @mark_cuda_test
    def test_pass_slice_to_kernel(self):
        mod = SourceModule("""
        __global__ void twice(float *a)
        {
          const int i = threadIdx.x + blockIdx.x * blockDim.x;
          a[i] *= 2;
        }
        """)

        multiply_them = mod.get_function("twice")

        a = np.ones(256**2, np.float32)
        a_gpu = gpuarray.to_gpu(a)

        multiply_them(a_gpu[256:-256], block=(256,1,1), grid=(254,1))

        a = a_gpu.get()
        assert (a[255:257]== np.array([1,2], np.float32)).all()
        assert (a[255*256-1:255*256+1] == np.array([2,1], np.float32)).all()

    @mark_cuda_test
    def test_scan(self):
        from pycuda.scan import ExclusiveScanKernel, InclusiveScanKernel
        for cls in [ExclusiveScanKernel, InclusiveScanKernel]:
            scan_kern = cls(np.int32, "a+b", "0")

            for n in [
                    10, 2**10-5, 2**10, 
                    2**20-2**18, 
                    2**20-2**18+5, 
                    2**10+5,
                    2**20+5,
                    2**20, 2**24
                    ]:
                host_data = np.random.randint(0, 10, n).astype(np.int32)
                gpu_data = gpuarray.to_gpu(host_data)

                scan_kern(gpu_data)

                desired_result = np.cumsum(host_data, axis=0)
                if cls is ExclusiveScanKernel:
                    desired_result -= host_data

                assert (gpu_data.get() == desired_result).all()

    @mark_cuda_test
    def test_stride_preservation(self):
        A = np.random.rand(3,3)
        AT = A.T
        print AT.flags.f_contiguous, AT.flags.c_contiguous
        AT_GPU = gpuarray.to_gpu(AT)
        print AT_GPU.flags.f_contiguous, AT_GPU.flags.c_contiguous
        assert np.allclose(AT_GPU.get(),AT)

    @mark_cuda_test
    def test_vector_fill(self):
        a_gpu = gpuarray.GPUArray(100, dtype=gpuarray.vec.float3)
        a_gpu.fill(gpuarray.vec.make_float3(0.0, 0.0, 0.0))
        a = a_gpu.get()
        assert a.dtype is gpuarray.vec.float3

    @mark_cuda_test
    def test_create_complex_zeros(self):
        gpuarray.zeros(3, np.complex64)

    @mark_cuda_test
    def test_reshape(self):
        a = np.arange(128).reshape(8, 16).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)

        # different ways to specify the shape
        a_gpu.reshape(4, 32)
        a_gpu.reshape((4, 32))
        a_gpu.reshape([4, 32])

    @mark_cuda_test
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
        view = a_gpu.view(np.float16)
        assert view.shape == (8, 32) and view.dtype == np.float16





if __name__ == "__main__":
    # make sure that import failures get reported, instead of skipping the tests.
    import pycuda.autoinit

    import sys
    if len(sys.argv) > 1:
        exec sys.argv[1]
    else:
        from py.test.cmdline import main
        main([__file__])
