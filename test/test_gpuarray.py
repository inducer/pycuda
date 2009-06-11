#! /usr/bin/env python
import pycuda.autoinit
import pycuda.driver as drv
import numpy
import numpy.linalg as la
import unittest
import pycuda.gpuarray as gpuarray
import sys
import test_abstract_array

class TestGPUArray(test_abstract_array.TestAbstractArray):                                      
    """tests the gpu array class"""

    def make_test_array(self,array):
        """creates a gpu array"""
        return gpuarray.to_gpu(array)

    def test_random(self):
        from pycuda.curandom import rand as curand
        for dtype in [numpy.float32, numpy.float64]:
            a = curand((10, 100), dtype=dtype).get()

            self.assert_((0 <= a).all())
            self.assert_((a < 1).all())

    def test_nan_arithmetic(self):
        def make_nan_contaminated_vector(size):
            shape = (size,)
            a = numpy.random.randn(*shape).astype(numpy.float32)
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

        for i in range(size):
            assert numpy.isnan(ab[i]) == numpy.isnan(ab_gpu[i])
        
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

    def test_take(self):
        idx = gpuarray.arange(0, 200000, 2, dtype=numpy.uint32)
        a = gpuarray.arange(0, 600000, 3, dtype=numpy.float32)
        result = gpuarray.take(a, idx)
        assert ((3*idx).get() == result.get()).all()

    def test_arange(self):
        a = gpuarray.arange(12, dtype=numpy.float32)

        res = a.get()

        for i in range(12):
            self.assert_(res[i] ==i)

    def test_reverse(self):
        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)
        a_cpu = self.make_test_array(a)

        a_cpu = a_cpu.reverse()


        b = a_cpu.get()

        for i in range(0,10):
            self.assert_(a[len(a)-1-i] == b[i])

    def test_sum(self):
        from pycuda.curandom import rand as curand
        a_gpu = curand((200000,))
        a = a_gpu.get()

        sum_a = numpy.sum(a)

        from pycuda.reduction import get_sum_kernel
        sum_a_gpu = gpuarray.sum(a_gpu).get()

        self.assert_(abs(sum_a_gpu-sum_a)/abs(sum_a) < 1e-4)

    def test_max(self):
        from pycuda.curandom import rand as curand

	a_gpu_d = curand((200000,), dtype=numpy.float64)
	a_gpu_s = curand((200000,), dtype=numpy.float32)
	a_gpu_i = curand((200000,), dtype=numpy.int32)

	a_d = a_gpu_d.get()
	a_s = a_gpu_s.get()
	a_i = a_gpu_i.get()

        max_a_d = numpy.max(a_d)
	max_a_s = numpy.max(a_s)
	max_a_i = numpy.max(a_i)

	from pycuda.reduction import get_max_kernel
	max_a_gpu_d = gpuarray.max(a_gpu_d, dtype=numpy.float64).get()
	max_a_gpu_s = gpuarray.max(a_gpu_s, dtype=numpy.float32).get()
	max_a_gpu_i = gpuarray.max(a_gpu_i, dtype=numpy.int32).get()


	self.assert_(abs(max_a_gpu_d-max_a_d)/abs(max_a_d) < 1e-4)
        self.assert_(abs(max_a_gpu_s-max_a_s)/abs(max_a_s) < 1e-4)
        self.assert_(abs(max_a_gpu_i-max_a_i)/abs(max_a_i) < 1e-4)

    def test_subset_max(self):
        from pycuda.curandom import rand as curand

        l_a = 200000
	gran = 5
        l_m = l_a - l_a / gran

	a_gpu_d = curand((l_a,), dtype=numpy.float64)
	a_gpu_s = curand((l_a,), dtype=numpy.float32)
	a_gpu_i = curand((l_a,), dtype=numpy.int32)

	a_d = a_gpu_d.get()
        a_s = a_gpu_s.get()
        a_i = a_gpu_i.get()

        meaningful_indices_gpu = gpuarray.zeros(l_m, dtype=numpy.int32)
	meaningful_indices = meaningful_indices_gpu.get()
        j = 0
	for i in range(len(meaningful_indices)):
	    meaningful_indices[i] = j
	    j = j + 1
	    if j % 5 == 0:
	        j = j + 1
        
	meaningful_indices_gpu = gpuarray.to_gpu(meaningful_indices)

        b_d = numpy.zeros(l_a)
        j = 0
        for i in meaningful_indices:
	    b_d[j] = a_d[i]
	    j = j + 1

        b_s = numpy.zeros(l_a)
        j = 0
        for i in meaningful_indices:
            b_s[j] = a_s[i]
            j = j + 1

        b_i = numpy.zeros(l_a)
        j = 0
        for i in meaningful_indices:
            b_i[j] = a_i[i]
            j = j + 1

        max_a_d = numpy.max(b_d)
        max_a_s = numpy.max(b_s)
        max_a_i = numpy.max(b_i)

	from pycuda.reduction import get_subset_max_kernel
	max_a_gpu_d = gpuarray.subset_max(meaningful_indices_gpu, a_gpu_d, 
	    dtype=numpy.float64).get()
        max_a_gpu_s = gpuarray.subset_max(meaningful_indices_gpu, a_gpu_s,
	            dtype=numpy.float32).get()
        max_a_gpu_i = gpuarray.subset_max(meaningful_indices_gpu, a_gpu_i,
	            dtype=numpy.int32).get()

	self.assert_(abs(max_a_gpu_d-max_a_d)/abs(max_a_d) < 1e-4)
        self.assert_(abs(max_a_gpu_s-max_a_s)/abs(max_a_s) < 1e-4)
        self.assert_(abs(max_a_gpu_i-max_a_i)/abs(max_a_i) < 1e-4)

    def test_min(self):
        from pycuda.curandom import rand as curand

	a_gpu_d = curand((200000,), dtype=numpy.float64)
	a_gpu_s = curand((200000,), dtype=numpy.float32)
	a_gpu_i = curand((200000,), dtype=numpy.int32)

	a_d = a_gpu_d.get()
	a_s = a_gpu_s.get()
	a_i = a_gpu_i.get()

        min_a_d = numpy.min(a_d)
	min_a_s = numpy.min(a_s)
	min_a_i = numpy.min(a_i)

	from pycuda.reduction import get_min_kernel
	min_a_gpu_d = gpuarray.min(a_gpu_d, dtype=numpy.float64).get()
        min_a_gpu_s = gpuarray.min(a_gpu_s, dtype=numpy.float32).get()
        min_a_gpu_i = gpuarray.min(a_gpu_i, dtype=numpy.int32).get()

	self.assert_(abs(min_a_gpu_d-min_a_d)/abs(min_a_d) < 1e-4)
	self.assert_(abs(min_a_gpu_s-min_a_s)/abs(min_a_s) < 1e-4)
	self.assert_(abs(min_a_gpu_i-min_a_i)/abs(min_a_i) < 1e-4)

    def test_subset_min(self):
        from pycuda.curandom import rand as curand

        l_a = 200000
	gran = 5
        l_m = l_a - l_a / gran

	a_gpu_d = curand((l_a,), dtype=numpy.float64)
	a_gpu_s = curand((l_a,), dtype=numpy.float32)
	a_gpu_i = curand((l_a,), dtype=numpy.int32)

	a_d = a_gpu_d.get()
	a_s = a_gpu_s.get()
	a_i = a_gpu_i.get()

        meaningful_indices_gpu = gpuarray.zeros(l_m, dtype=numpy.int32)
	meaningful_indices = meaningful_indices_gpu.get()
        j = 0
	for i in range(len(meaningful_indices)):
	    meaningful_indices[i] = j
	    j = j + 1
	    if j % 5 == 0:
	        j = j + 1
        
	meaningful_indices_gpu = gpuarray.to_gpu(meaningful_indices)

        b_d = numpy.zeros(l_a)
	b_d.fill(1)
        j = 0
        for i in meaningful_indices:
	    b_d[j] = a_d[i]
	    j = j + 1

        b_s = numpy.zeros(l_a)
	b_s.fill(1)
        j = 0
        for i in meaningful_indices:
	    b_s[j] = a_s[i]
	    j = j + 1

        b_i = numpy.zeros(l_a)
	b_i.fill(1E10)
        j = 0
        for i in meaningful_indices:
	    b_i[j] = a_i[i]
	    j = j + 1

        min_a_d = numpy.min(b_d)
	min_a_s = numpy.min(b_s)
	min_a_i = numpy.min(b_i)

	from pycuda.reduction import get_subset_min_kernel
	min_a_gpu_d = gpuarray.subset_min(meaningful_indices_gpu, a_gpu_d, 
	    dtype=numpy.float64).get()
	min_a_gpu_s = gpuarray.subset_min(meaningful_indices_gpu, a_gpu_s, 
	    dtype=numpy.float32).get()
	min_a_gpu_i = gpuarray.subset_min(meaningful_indices_gpu, a_gpu_i, 
	    dtype=numpy.int32).get()

	self.assert_(abs(min_a_gpu_d-min_a_d)/abs(min_a_d) < 1e-4)
	self.assert_(abs(min_a_gpu_s-min_a_s)/abs(min_a_s) < 1e-4)
	self.assert_(abs(min_a_gpu_i-min_a_i)/abs(min_a_i) < 1e-4)

    def test_dot(self):
        from pycuda.curandom import rand as curand
        a_gpu = curand((200000,))
        a = a_gpu.get()
        b_gpu = curand((200000,))
        b = b_gpu.get()

        dot_ab = numpy.dot(a, b)

        dot_ab_gpu = gpuarray.dot(a_gpu, b_gpu).get()

        self.assert_(abs(dot_ab_gpu-dot_ab)/abs(dot_ab) < 1e-4)

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

            self.assert_(la.norm(a_gpu_slice.get()-a_slice) == 0)

if __name__ == '__main__':
    unittest.main()
