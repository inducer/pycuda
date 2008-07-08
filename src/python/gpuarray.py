from __future__ import division
import numpy
from pytools import memoize
import pycuda.driver as drv



def splay(n, min_threads=None, max_threads=128, max_blocks=80):
    # stolen from cublas

    if min_threads is None:
        min_threads = WARP_SIZE

    if n < min_threads:
        block_count = 1
        elems_per_block = n
        threads_per_block = min_threads
    elif n < (max_blocks * min_threads):
        block_count = (n + min_threads - 1) // min_threads
        threads_per_block = min_threads
        elems_per_block = threads_per_block
    elif n < (max_blocks * max_threads):
        block_count = max_blocks
        grp = (n + min_threads - 1) // min_threads
        threads_per_block = ((grp + max_blocks -1) // max_blocks) * min_threads
        elems_per_block = threads_per_block
    else:
        block_count = max_blocks
        threads_per_block = max_threads
        grp = (n + min_threads - 1) // min_threads
        grp = (grp + max_blocks - 1) // max_blocks
        elems_per_block = grp * min_threads

    #print "bc:%d tpb:%d epb:%d" % (block_count, threads_per_block, elems_per_block)
    return block_count, threads_per_block, elems_per_block




NVCC_OPTIONS = []




@memoize
def get_axpbyz_kernel():
    mod = drv.SourceModule("""
        __global__ void axpbyz(float a, float *x, float b, float *y, float *z,
          int n)
        {
          int tid = threadIdx.x;
          int total_threads = gridDim.x*blockDim.x;
          int cta_start = blockDim.x*blockIdx.x;
          int i;
                
          for (i = cta_start + tid; i < n; i += total_threads) 
          {
            z[i] = a*x[i] + b*y[i];
          }
        }
        """,
        options=NVCC_OPTIONS)

    return mod.get_function("axpbyz")


@memoize
def get_add_kernel():
    mod = drv.SourceModule("""
        __global__ void add(float a, float *x,float *z,
          int n)
        {
          int tid = threadIdx.x;
          int total_threads = gridDim.x*blockDim.x;
          int cta_start = blockDim.x*blockIdx.x;
          int i;

          for (i = cta_start + tid; i < n; i += total_threads)
          {
            z[i] = x[i] + a;
          }

        }
        """,
        options=NVCC_OPTIONS)

    return mod.get_function("add")

@memoize
def get_subr_kernel():
    mod = drv.SourceModule("""
        __global__ void subr(float a, float *x,float *z,
          int n)
        {
          int tid = threadIdx.x;
          int total_threads = gridDim.x*blockDim.x;
          int cta_start = blockDim.x*blockIdx.x;
          int i;

          for (i = cta_start + tid; i < n; i += total_threads)
          {
            z[i] = a - x[i];
          }

        }
        """,
        options=NVCC_OPTIONS)

    return mod.get_function("subr")


@memoize
def get_sub_kernel():
    mod = drv.SourceModule("""
        __global__ void sub(float a, float *x,float *z,
          int n)
        {
          int tid = threadIdx.x;
          int total_threads = gridDim.x*blockDim.x;
          int cta_start = blockDim.x*blockIdx.x;
          int i;

          for (i = cta_start + tid; i < n; i += total_threads)
          {
            z[i] = x[i] - a;
          }

        }
        """,
        options=NVCC_OPTIONS)

    return mod.get_function("sub")

@memoize
def get_multiply_kernel():
    mod = drv.SourceModule("""
        __global__ void multiply(float *x, float *y, float *z,
          int n)
        {
          int tid = threadIdx.x;
          int total_threads = gridDim.x*blockDim.x;
          int cta_start = blockDim.x*blockIdx.x;
          int i;
                
          for (i = cta_start + tid; i < n; i += total_threads) 
          {
            z[i] = x[i] * y[i];
          }
        }
        """,
        options=NVCC_OPTIONS)

    return mod.get_function("multiply")

@memoize
def get_divide_kernel():
    mod = drv.SourceModule("""
        __global__ void divide(float *x, float *y, float *z,
          int n)
        {
          int tid = threadIdx.x;
          int total_threads = gridDim.x*blockDim.x;
          int cta_start = blockDim.x*blockIdx.x;
          int i;

          for (i = cta_start + tid; i < n; i += total_threads)
          {
            z[i] = x[i] / y[i];
          }
        }
        """,
        options=NVCC_OPTIONS)

    return mod.get_function("divide")

@memoize
def get_divide_scalar_kernel():
    mod = drv.SourceModule("""
        __global__ void divideScalar(float *x, float y, float *z,
          int n)
        {
          int tid = threadIdx.x;
          int total_threads = gridDim.x*blockDim.x;
          int cta_start = blockDim.x*blockIdx.x;
          int i;

          for (i = cta_start + tid; i < n; i += total_threads)
          {
            z[i] = x[i] / y;
          }
        }
        """,
        options=NVCC_OPTIONS)

    return mod.get_function("divideScalar")

@memoize
def get_divider_scalar_kernel():
    mod = drv.SourceModule("""
        __global__ void divideRScalar(float *x, float y, float *z,
          int n)
        {
          int tid = threadIdx.x;
          int total_threads = gridDim.x*blockDim.x;
          int cta_start = blockDim.x*blockIdx.x;
          int i;

          for (i = cta_start + tid; i < n; i += total_threads)
          {
            z[i] = y / x[i];
          }
        }
        """,
        options=NVCC_OPTIONS)

    return mod.get_function("divideRScalar")





@memoize
def get_scale_kernel():
    mod = drv.SourceModule("""
        __global__ void scale(float a, float *x, float *y,int n)
        {
          int tid = threadIdx.x;
          int total_threads = gridDim.x*blockDim.x;
          int cta_start = blockDim.x*blockIdx.x;
          int i;
                
          for (i = cta_start + tid; i < n; i += total_threads) 
          {
            y[i] = a*x[i];
          }
        }
        """,
        options=NVCC_OPTIONS)

    return mod.get_function("scale")




@memoize
def get_fill_kernel():
    mod = drv.SourceModule("""
        __global__ void fill(float a, float *x, int n)
        {
          int tid = threadIdx.x;
          int total_threads = gridDim.x*blockDim.x;
          int cta_start = blockDim.x*blockIdx.x;
          int i;
                
          for (i = cta_start + tid; i < n; i += total_threads) 
          {
            x[i] = a;
          }
        }
        """,
        options=NVCC_OPTIONS)

    return mod.get_function("fill")




WARP_SIZE = 32



"""

    Defines a GPUArray which is used todo array based calculation on the GPU. For this reason we overwrite most
    of the operators to make it as easy as possible to work with this object

"""
class GPUArray(object):
    
    
    """
       Constructor for this class to create an array of the given shape
    """
    def __init__(self, shape, dtype, stream=None):
        self.shape = shape
        self.dtype = numpy.dtype(dtype)
        from pytools import product
        self.size = product(shape)
        if self.size:
            self.gpudata = drv.mem_alloc(self.size * self.dtype.itemsize)
        else:
            self.gpudata = None
        self.stream = stream


    """

       Compiles all the defined kernels to save time. Mostly needed for benchmarks, since you don't
       want to time the compiling overhead

    """
    @staticmethod
    def compile_kernels():
        # useful for benchmarking
        get_axpbyz_kernel()
        get_scale_kernel()
        get_fill_kernel()
        get_multiply_kernel()
        get_add_kernel()
        get_sub_kernel()
        get_subr_kernel()
        get_divide_kernel()
        get_divide_scalar_kernel()
        get_divider_scalar_kernel()


    """
    
       Set array content. This copies the given array to the gpu
    
    """
    def set(self, ary, stream=None):
        assert ary.size == self.size
        assert ary.dtype == self.dtype
        if self.size:
            drv.memcpy_htod(self.gpudata, ary, stream)


    """
    
        Get the array content from the device
    
    """
    def get(self, ary=None, stream=None, pagelocked=False):
        if ary is None:
            if pagelocked:
                ary = drv.pagelocked_empty(self.shape, self.dtype)
            else:
                ary = numpy.empty(self.shape, self.dtype)
        else:
            assert ary.size == self.size
            assert ary.dtype == self.dtype
        if self.size:
            drv.memcpy_dtoh(ary, self.gpudata)
        return ary

    def __str__(self):
        return str(self.get())

    def __repr__(self):
        return repr(self.get())

    def _axpbyz(self, selffac, other, otherfac, out):
        assert self.dtype == numpy.float32
        assert self.shape == other.shape
        assert self.dtype == other.dtype

        if self.stream is not None or other.stream is not None:
            assert self.stream is other.stream

        block_count, threads_per_block, elems_per_block = splay(self.size, WARP_SIZE, 128, 80)

        get_axpbyz_kernel()(numpy.float32(selffac), self.gpudata, 
                numpy.float32(otherfac), other.gpudata, 
                out.gpudata, numpy.int32(self.size),
                block=(threads_per_block,1,1), grid=(block_count,1),
                stream=self.stream)

        return out


    """
    
       Add an array with a scalar and store the result in out array
       
    """
    def _add(self, other, out):
        assert self.dtype == numpy.float32

        block_count, threads_per_block, elems_per_block = splay(self.size, WARP_SIZE, 128, 80)

        get_add_kernel()(
                numpy.float32(other),
                self.gpudata,
                out.gpudata, numpy.int32(self.size),
                block=(threads_per_block,1,1), grid=(block_count,1),
                stream=self.stream)

        return out


    """

       Divides an array by a scalar

       y = self / n

    """
    def _divScalar(self, other, out):
        assert self.dtype == numpy.float32

        block_count, threads_per_block, elems_per_block = splay(self.size, WARP_SIZE, 128, 80)

        get_divide_scalar_kernel()(
                self.gpudata,
                numpy.float32(other),
                out.gpudata, numpy.int32(self.size),
                block=(threads_per_block,1,1), grid=(block_count,1),
                stream=self.stream)

        return out


    """

       Divides an array by a scalar
      
       y = n / self

    """
    def _divRScalar(self, other, out):
        assert self.dtype == numpy.float32

        block_count, threads_per_block, elems_per_block = splay(self.size, WARP_SIZE, 128, 80)

        get_divider_scalar_kernel()(
                self.gpudata,
                numpy.float32(other),
                out.gpudata, numpy.int32(self.size),
                block=(threads_per_block,1,1), grid=(block_count,1),
                stream=self.stream)

        return out


    """
    
       Divides an array by another array.

    """
    def _div(self, other, out):
        assert self.dtype == numpy.float32
        assert self.shape == other.shape
        assert self.dtype == other.dtype

        block_count, threads_per_block, elems_per_block = splay(self.size, WARP_SIZE, 128, 80)

        get_divide_kernel()(self.gpudata, other.gpudata,
                out.gpudata, numpy.int32(self.size),
                block=(threads_per_block,1,1), grid=(block_count,1),
                stream=self.stream)

        return out


    """
    
       Substract a scalar from an array
       
    """
    def _sub(self, other, out):
        assert self.dtype == numpy.float32

        block_count, threads_per_block, elems_per_block = splay(self.size, WARP_SIZE, 128, 80)

        get_sub_kernel()(
                numpy.float32(other),
                self.gpudata,
                out.gpudata, numpy.int32(self.size),
                block=(threads_per_block,1,1), grid=(block_count,1),
                stream=self.stream)

        return out


    """

       Substract an array from a scalar

    """
    def _subr(self, other, out):
        assert self.dtype == numpy.float32

        block_count, threads_per_block, elems_per_block = splay(self.size, WARP_SIZE, 128, 80)

        get_subr_kernel()(
                numpy.float32(other),
                self.gpudata,
                out.gpudata, numpy.int32(self.size),
                block=(threads_per_block,1,1), grid=(block_count,1),
                stream=self.stream)

        return out


    """
    
       Add an array with an array or an array with a scalar
       
    """
    def __add__(self, other):
        if isinstance(other, (int, float, complex)):
            # add a scalar
            if other == 0:
                return self
            else:
                #scalar addition
                result = GPUArray(self.shape, self.dtype)
                return self._add(other,result)
        else:
            # add another vector
            result = GPUArray(self.shape, self.dtype)
            return self._axpbyz(1, other, 1, result)

    __radd__ = __add__


    """
    
       Substract an array from an array or a scalar from an array 
    
    """
    def __sub__(self, other):
        if isinstance(other, (int, float, complex)):
            #if array - 0 than just return the array since its the same anyway
            if other == 0:
                return self
            else:
                #create a new array for the result
                result = GPUArray(self.shape, self.dtype)
                return self._sub(other, result)
        else:
            result = GPUArray(self.shape, self.dtype)
            return self._axpbyz(1, other, -1, result)


    """

       Divides an array by an array or a scalar

       x = self / n

    """
    def __div__(self, other):
        if isinstance(other, (int, float, complex)):
            #if array - 0 than just return the array since its the same anyway
            if other == 0:
                return self
            else:
                #create a new array for the result
                result = GPUArray(self.shape, self.dtype)
                return self._divScalar(other, result)
        else:
            result = GPUArray(self.shape, self.dtype)
            return self._div(other, result)


    """

       Divides an array by a scalar or an array

       x = n / self

    """
    def __rdiv__(self,other):
        if isinstance(other, (int, float, complex)):
            #if array - 0 than just return the array since its the same anyway
            if other == 0:
                return self
            else:
                #create a new array for the result
                result = GPUArray(self.shape, self.dtype)
                return self._divRScalar(other, result)
        else:
            result = GPUArray(self.shape, self.dtype)

            assert self.dtype == numpy.float32
            assert self.shape == other.shape
            assert self.dtype == other.dtype

            block_count, threads_per_block, elems_per_block = splay(self.size, WARP_SIZE, 128, 80)

            get_divide_kernel()(other.gpudata, self.gpudata,
                    out.gpudata, numpy.int32(self.size),
                    block=(threads_per_block,1,1), grid=(block_count,1),
                    stream=self.stream)

            return out


    """

       Substracts an array by a scalar or an array

       x = n - self

    """
    def __rsub__(self,other):
        if isinstance(other, (int, float, complex)):
            #if array - 0 than just return the array since its the same anyway
            if other == 0:
                return self
            else:
                #create a new array for the result
                result = GPUArray(self.shape, self.dtype)
                return self._subr(other, result)
        else:
            result = GPUArray(self.shape, self.dtype)

            assert self.dtype == numpy.float32
            assert self.shape == other.shape
            assert self.dtype == other.dtype

            block_count, threads_per_block, elems_per_block = splay(self.size, WARP_SIZE, 128, 80)

            get_axpbyz_kernel()(numpy.float32(-1), other.gpudata,
                    numpy.float32(-1), self.gpudata,
                    out.gpudata, numpy.int32(self.size),
                    block=(threads_per_block,1,1), grid=(block_count,1),
                    stream=self.stream)

            return out


    def __iadd__(self, other):
        return self._axpbyz(1, other, 1, self)

    def __isub__(self, other):
        return self._axpbyz(1, other, -1, self)

    def _scale(self, factor, out):
        assert self.dtype == numpy.float32

        block_count, threads_per_block, elems_per_block = splay(self.size, WARP_SIZE, 128, 80)

        get_scale_kernel()(numpy.float32(factor), self.gpudata, 
                out.gpudata, numpy.int32(self.size),
                block=(threads_per_block,1,1), grid=(block_count,1),
                stream=self.stream)

        return out

    def _elwise_multiply(self, other, out):
        assert self.dtype == numpy.float32
        assert self.dtype == numpy.float32

        block_count, threads_per_block, elems_per_block = splay(self.size, WARP_SIZE, 128, 80)

        get_multiply_kernel()(
                self.gpudata, other.gpudata,
                out.gpudata, numpy.int32(self.size),
                block=(threads_per_block,1,1), grid=(block_count,1),
                stream=self.stream)

        return out

    def __neg__(self):
        result = GPUArray(self.shape, self.dtype)
        return self._scale(-1, result)

    def __mul__(self, other):
        result = GPUArray(self.shape, self.dtype)
        if isinstance(other, (int, float, complex)):
            return self._scale(other, result)
        else:
            return self._elwise_multiply(other, result)

    def __rmul__(self, scalar):
        result = GPUArray(self.shape, self.dtype)
        return self._scale(scalar, result)

    def __imul__(self, scalar):
        return self._scale(scalar, self)


    """

       fills the array with values

    """
    def fill(self, value):
        assert self.dtype == numpy.float32

        block_count, threads_per_block, elems_per_block = splay(self.size, WARP_SIZE, 128, 80)

        get_fill_kernel()(numpy.float32(value), self.gpudata, numpy.int32(self.size),
                block=(threads_per_block,1,1), grid=(block_count,1),
                stream=self.stream)

        return self


    def bind_to_texref(self, texref):
        texref.set_address(self.gpudata, self.size*self.dtype.itemsize)




"""

    converts a numpy array to a gpu array

"""
def to_gpu(ary, stream=None):
    result = GPUArray(ary.shape, ary.dtype)
    result.set(ary, stream)
    return result


empty = GPUArray


"""

   creats an gpu array of the given size filled with zeros

"""
def zeros(shape, dtype, stream=None):
    result = GPUArray(shape, dtype, stream)
    result.fill(0)
    return result


