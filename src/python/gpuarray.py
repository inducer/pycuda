from __future__ import division
import numpy
import pycuda.elementwise as elementwise
import random as random
from pytools import memoize
from pytools.diskdict import DiskDict
import pycuda.driver as drv



@memoize
def _splay_backend(n, dev):
    # heavily modified from cublas
    from pycuda.tools import DeviceData
    devdata = DeviceData(dev)

    min_threads = devdata.warp_size
    max_threads = 128
    max_blocks = 4 * devdata.thread_blocks_per_mp \
            * dev.get_attribute(drv.device_attribute.MULTIPROCESSOR_COUNT)

    if n < min_threads:
        block_count = 1
        threads_per_block = min_threads
    elif n < (max_blocks * min_threads):
        block_count = (n + min_threads - 1) // min_threads
        threads_per_block = min_threads
    elif n < (max_blocks * max_threads):
        block_count = max_blocks
        grp = (n + min_threads - 1) // min_threads
        threads_per_block = ((grp + max_blocks -1) // max_blocks) * min_threads
    else:
        block_count = max_blocks
        threads_per_block = max_threads

    #print "n:%d bc:%d tpb:%d" % (n, block_count, threads_per_block)
    return (block_count, 1), (threads_per_block, 1, 1)




def splay(n, dev=None):
    if dev is None:
        dev = drv.Context.get_device()
    return _splay_backend(n, dev)




class GPUArray(object): 
    """A GPUArray is used to do array based calculation on the GPU. 

    This is mostly supposed to be a numpy-workalike. Operators
    work on an element-by-element basis, just like numpy.ndarray.
    """

    def __init__(self, shape, dtype, stream=None, allocator=drv.mem_alloc):
        self.shape = shape
        self.dtype = numpy.dtype(dtype)

        s = 1
        for dim in shape:
            s *= dim
        self.mem_size = self.size = s
        self.nbytes = self.dtype.itemsize * self.size

        self.allocator = allocator
        if self.size:
            self.gpudata = self.allocator(self.size * self.dtype.itemsize)
        else:
            self.gpudata = None
        self.stream = stream

        self._grid, self._block = splay(self.mem_size)

    @classmethod
    def compile_kernels(cls):
        # useful for benchmarking
        elementwise._compile_kernels(cls)

    def set(self, ary, stream=None):
        assert ary.size == self.size
        assert ary.dtype == self.dtype
        if self.size:
            drv.memcpy_htod(self.gpudata, ary, stream)

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

    # kernel invocation wrappers ----------------------------------------------
    def _axpbyz(self, selffac, other, otherfac, out, add_timer=None):
        """Compute ``out = selffac * self + otherfac*other``, 
        where `other` is a vector.."""
        assert self.dtype == numpy.float32
        assert self.shape == other.shape
        assert self.dtype == other.dtype

        func = elementwise.get_axpbyz_kernel()
        func.set_block_shape(*self._block)

        if add_timer is not None:
            add_timer(3*self.size, func.prepared_timed_call(self._grid, 
                selffac, self.gpudata, otherfac, other.gpudata, 
                out.gpudata, self.mem_size))
        else:
            func.prepared_async_call(self._grid, self.stream,
                    selffac, self.gpudata, otherfac, other.gpudata, 
                    out.gpudata, self.mem_size)

        return out

    def _axpbz(self, selffac, other, out):
        """Compute ``out = selffac * self + other``, where `other` is a scalar."""
        assert self.dtype == numpy.float32

        func = elementwise.get_axpbz_kernel()
        func.set_block_shape(*self._block)
        func.prepared_async_call(self._grid, self.stream,
                selffac, self.gpudata,
                other, out.gpudata, self.mem_size)

        return out

    def _elwise_multiply(self, other, out):
        assert self.dtype == numpy.float32
        assert self.dtype == numpy.float32

        func = elementwise.get_multiply_kernel()
        func.set_block_shape(*self._block)
        func.prepared_async_call(self._grid, self.stream,
                self.gpudata, other.gpudata,
                out.gpudata, self.mem_size)

        return out

    def _rdiv_scalar(self, other, out):
        """Divides an array by a scalar::
          
           y = n / self 
        """

        assert self.dtype == numpy.float32

        func = elementwise.get_rdivide_elwise_kernel()
        func.set_block_shape(*self._block)
        func.prepared_async_call(self._grid, self.stream,
                self.gpudata, other,
                out.gpudata, self.mem_size)

        return out

    def _div(self, other, out):
        """Divides an array by another array."""

        assert self.dtype == numpy.float32
        assert self.shape == other.shape
        assert self.dtype == other.dtype

        func = elementwise.get_divide_kernel()
        func.set_block_shape(*self._block)
        func.prepared_async_call(self._grid, self.stream,
                self.gpudata, other.gpudata,
                out.gpudata, self.mem_size)

        return out

    def _new_like_me(self):
        return self.__class__(self.shape, self.dtype, allocator=self.allocator)

    # operators ---------------------------------------------------------------
    def mul_add(self, selffac, other, otherfac, add_timer=None):
        """Return `selffac * self + otherfac*other`.
        """
        result = self._new_like_me()
        return self._axpbyz(selffac, other, otherfac, result, add_timer)

    def __add__(self, other):
        """Add an array with an array or an array with a scalar."""

        if isinstance(other, GPUArray):
            # add another vector
            result = self._new_like_me()
            return self._axpbyz(1, other, 1, result)
        else:
            # add a scalar
            if other == 0:
                return self
            else:
                result = self._new_like_me()
                return self._axpbz(1, other, result)

    __radd__ = __add__

    def __sub__(self, other):
        """Substract an array from an array or a scalar from an array."""

        if isinstance(other, GPUArray):
            result = self._new_like_me()
            return self._axpbyz(1, other, -1, result)
        else:
            if other == 0:
                return self
            else:
                # create a new array for the result
                result = self._new_like_me()
                return self._axpbz(1, -other, result)

    def __rsub__(self,other):
        """Substracts an array by a scalar or an array:: 

           x = n - self
        """
        # other must be a scalar
        result = self._new_like_me()
        return self._axpbz(-1, other, result)

    def __iadd__(self, other):
        return self._axpbyz(1, other, 1, self)

    def __isub__(self, other):
        return self._axpbyz(1, other, -1, self)

    def __neg__(self):
        result = self._new_like_me()
        return self._axpbz(-1, 0, result)

    def __mul__(self, other):
        result = self._new_like_me()
        if isinstance(other, GPUArray):
            return self._elwise_multiply(other, result)
        else:
            return self._axpbz(other, 0, result)

    def __rmul__(self, scalar):
        result = self._new_like_me()
        return self._axpbz(scalar, 0, result)

    def __imul__(self, scalar):
        return self._axpbz(scalar, 0, self)

    def __div__(self, other):
        """Divides an array by an array or a scalar::

           x = self / n
        """
        if isinstance(other, GPUArray):
            result = self._new_like_me()
            return self._div(other, result)
        else:
            if other == 1:
                return self
            else:
                # create a new array for the result
                result = self._new_like_me()
                return self._axpbz(1/other, 0, result)

    __truediv__ = __div__

    def __rdiv__(self,other):
        """Divides an array by a scalar or an array::

           x = n / self
        """

        if isinstance(other, GPUArray):
            result = self._new_like_me()

            func = elementwise.get_divide_kernel()
            func.set_block_shape(*self._block)
            func.prepared_async_call(self._grid, self.stream,
                    other.gpudata, self.gpudata, out.gpudata, 
                    self.mem_size)

            return result
        else:
            if other == 1:
                return self
            else:
                # create a new array for the result
                result = self._new_like_me()
                return self._rdiv_scalar(other, result)

    __rtruediv__ = __div__

    def fill(self, value):
        """fills the array with the specified value"""
        assert self.dtype == numpy.float32

        func = elementwise.get_fill_kernel()
        func.set_block_shape(*self._block)
        func.prepared_async_call(self._grid, self.stream,
                value, self.gpudata, self.mem_size)

        return self

    def bind_to_texref(self, texref):
        texref.set_address(self.gpudata, self.size*self.dtype.itemsize)

    def __len__(self):
        """returns the len of the internal array"""
        return self.size

    def __abs__(self):
        """Return a `GPUArray` of the absolute values of the elements
        of `self`.
        """

        assert self.dtype == numpy.float32

        result = GPUArray(self.shape, self.dtype)

        func = elementwise.get_unary_func_kernel("fabs")
        func.set_block_shape(*self._block)
        func.prepared_async_call(self._grid, self.stream,
                self.gpudata,result.gpudata, self.mem_size)

        return result

    def __pow__(self, other):
        """pow function::
 
           example:
                   array = pow(array)
                   array = pow(array,4)
                   array = pow(array,array)

        """
        result = GPUArray(self.shape, self.dtype)

        if isinstance(other, GPUArray):
            assert self.shape == other.shape
            assert self.dtype == other.dtype

            func = elementwise.get_pow_array_kernel()
            func.set_block_shape(*self._block)
            func.prepared_async_call(self._grid, self.stream,
                    self.gpudata, other.gpudata, result.gpudata,
                    self.mem_size)
            
            return result
        else:
            func = elementwise.get_pow_kernel()
            func.set_block_shape(*self._block)
            func.prepared_async_call(self._grid, self.stream,
                    other, self.gpudata, result.gpudata,
                    self.mem_size)

            return result

    def reverse(self):
        """Return this array in reversed order. The array is treated
        as one-dimensional.
        """

        assert self.dtype == numpy.float32
        
        result = GPUArray(self.shape, self.dtype)

        func = elementwise.get_reverse_kernel()
        func.set_block_shape(*self._block)
        func.prepared_async_call(self._grid, self.stream,
                self.gpudata, result.gpudata,
                self.mem_size)

        return result




def to_gpu(ary, stream=None, allocator=drv.mem_alloc):
    """converts a numpy array to a GPUArray"""
    result = GPUArray(ary.shape, ary.dtype, stream, allocator)
    result.set(ary, stream)
    return result

empty = GPUArray

def zeros(shape, dtype, stream=None, allocator=drv.mem_alloc):
    """Returns an array of the given shape and dtype filled with 0's."""

    result = GPUArray(shape, dtype, stream, allocator)
    result.fill(0)
    return result

def empty_like(other_ary):
    result = GPUArray(
            other_ary.shape, other_ary.dtype, 
            other_ary.stream, other_ary.allocator)
    return result

def zeros_like(other_ary):
    result = GPUArray(
            other_ary.shape, other_ary.dtype, 
            other_ary.stream, other_ary.allocator)
    result.fill(0)
    return result


def arange(*args, **kwargs):
    """Create an array filled with numbers spaced `step` apart,
    starting from `start` and ending at `stop`.
    
    For floating point arguments, the length of the result is
    `ceil((stop - start)/step)`.  This rule may result in the last
    element of the result being greater than stop.
    """

    # argument processing -----------------------------------------------------

    # Yuck. Thanks, numpy developers. ;)

    start = None
    stop = None
    step = None
    dtype = None

    if isinstance(args[-1], numpy.dtype):
        dtype = args[-1]
        args = args[:-1]

    argc = len(args)
    if argc == 0:
        raise ValueError, "stop argument required"
    elif argc == 1:
        stop = args[0]
    elif argc == 2:
        start = args[0]
        stop = args[1]
    elif argc == 3:
        start = args[0]
        stop = args[1]
        step = args[2]
    else:
        raise ValueError, "too many arguments"

    admissible_names = ["start", "stop", "step", "dtype"]
    for k, v in kwargs.iteritems():
        if k in admissible_names:
            if locals()[k] is None:
                locals()[k] = v
            else:
                raise ValueError, "may not specify 'dtype' by position and keyword" % k
        else:
            raise ValueError, "unexpected keyword argument '%s'" % k

    if start is None:
        start = 0
    if step is None:
        step = 1
    if dtype is None:
        #dtype = numpy.array([start, stop, step]).dtype
        dtype = numpy.float32

    # actual functionality ----------------------------------------------------
    assert dtype == numpy.float32

    dtype = numpy.dtype(dtype)
    start = dtype.type(start)
    step = dtype.type(step)

    from math import ceil
    size = int(ceil((stop-start)/step))
  
    result = GPUArray((size,), dtype)

    func = elementwise.get_arange_kernel()
    func.set_block_shape(*result._block)
    func.prepared_async_call(result._grid, result.stream,
            result.gpudata, start, step, size)

    return result

