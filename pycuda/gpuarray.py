from __future__ import division
import numpy
import pycuda.elementwise as elementwise
from pytools import memoize
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




def _get_common_dtype(obj1, obj2):
    return (obj1.dtype.type(0) + obj2.dtype.type(0)).dtype




class GPUArray(object): 
    """A GPUArray is used to do array-based calculation on the GPU. 

    This is mostly supposed to be a numpy-workalike. Operators
    work on an element-by-element basis, just like numpy.ndarray.
    """

    def __init__(self, shape, dtype, allocator=drv.mem_alloc, 
            base=None, gpudata=None):
        try:
            s = 1
            for dim in shape:
                s *= dim
        except TypeError:
            assert isinstance(shape, int)
            s = shape
            shape = (shape,)

        self.shape = shape
        self.dtype = numpy.dtype(dtype)

        self.mem_size = self.size = s
        self.nbytes = self.dtype.itemsize * self.size

        self.allocator = allocator
        if gpudata is None:
            if self.size:
                self.gpudata = self.allocator(self.size * self.dtype.itemsize)
            else:
                self.gpudata = None
            
            assert base is None
        else:
            self.gpudata = gpudata

        self.base = base

        self._grid, self._block = splay(self.mem_size)

    def set(self, ary):
        assert ary.size == self.size
        assert ary.dtype == self.dtype
        if self.size:
            drv.memcpy_htod(self.gpudata, ary)

    def set_async(self, ary, stream=None):
        assert ary.size == self.size
        assert ary.dtype == self.dtype
        if self.size:
            drv.memcpy_htod_async(self.gpudata, ary, stream)

    def get(self, ary=None, pagelocked=False):
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

    def get_async(self, stream=None, ary=None):
        if ary is None:
            ary = drv.pagelocked_empty(self.shape, self.dtype)
        else:
            assert ary.size == self.size
            assert ary.dtype == self.dtype

        if self.size:
            drv.memcpy_dtoh_async(ary, self.gpudata, stream)
        return ary

    def __str__(self):
        return str(self.get())

    def __repr__(self):
        return repr(self.get())

    def __hash__(self):
        raise TypeError("GPUArrays are not hashable.")

    # kernel invocation wrappers ----------------------------------------------
    def _axpbyz(self, selffac, other, otherfac, out, add_timer=None, stream=None):
        """Compute ``out = selffac * self + otherfac*other``, 
        where `other` is a vector.."""
        assert self.shape == other.shape

        func = elementwise.get_axpbyz_kernel(self.dtype, other.dtype, out.dtype)
        func.set_block_shape(*self._block)

        if add_timer is not None:
            add_timer(3*self.size, func.prepared_timed_call(self._grid, 
                selffac, self.gpudata, otherfac, other.gpudata, 
                out.gpudata, self.mem_size))
        else:
            func.prepared_async_call(self._grid, stream,
                    selffac, self.gpudata, otherfac, other.gpudata, 
                    out.gpudata, self.mem_size)

        return out

    def _axpbz(self, selffac, other, out, stream=None):
        """Compute ``out = selffac * self + other``, where `other` is a scalar."""
        func = elementwise.get_axpbz_kernel(self.dtype)
        func.set_block_shape(*self._block)
        func.prepared_async_call(self._grid, stream,
                selffac, self.gpudata,
                other, out.gpudata, self.mem_size)

        return out

    def _elwise_multiply(self, other, out, stream=None):
        func = elementwise.get_multiply_kernel(self.dtype, other.dtype, out.dtype)
        func.set_block_shape(*self._block)
        func.prepared_async_call(self._grid, stream,
                self.gpudata, other.gpudata,
                out.gpudata, self.mem_size)

        return out

    def _rdiv_scalar(self, other, out, stream=None):
        """Divides an array by a scalar::
          
           y = n / self 
        """

        assert self.dtype == numpy.float32

        func = elementwise.get_rdivide_elwise_kernel(self.dtype)
        func.set_block_shape(*self._block)
        func.prepared_async_call(self._grid, stream,
                self.gpudata, other,
                out.gpudata, self.mem_size)

        return out

    def _div(self, other, out, stream=None):
        """Divides an array by another array."""

        assert self.shape == other.shape

        func = elementwise.get_divide_kernel(self.dtype, other.dtype, out.dtype)
        func.set_block_shape(*self._block)
        func.prepared_async_call(self._grid, stream,
                self.gpudata, other.gpudata,
                out.gpudata, self.mem_size)

        return out

    def _new_like_me(self, dtype=None):
        if dtype is None:
            dtype = self.dtype
        return self.__class__(self.shape, dtype,
                allocator=self.allocator)

    # operators ---------------------------------------------------------------
    def mul_add(self, selffac, other, otherfac, add_timer=None, stream=None):
        """Return `selffac * self + otherfac*other`.
        """
        result = self._new_like_me(_get_common_dtype(self, other))
        return self._axpbyz(selffac, other, otherfac, result, add_timer)

    def __add__(self, other):
        """Add an array with an array or an array with a scalar."""

        if isinstance(other, GPUArray):
            # add another vector
            result = self._new_like_me(_get_common_dtype(self, other))
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
            result = self._new_like_me(_get_common_dtype(self, other))
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
        if isinstance(other, GPUArray):
            result = self._new_like_me(_get_common_dtype(self, other))
            return self._elwise_multiply(other, result)
        else:
            result = self._new_like_me()
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
            result = self._new_like_me(_get_common_dtype(self, other))
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
            result = self._new_like_me(_get_common_dtype(self, other))

            func = elementwise.get_divide_kernel()
            func.set_block_shape(*self._block)
            func.prepared_async_call(self._grid, None,
                    other.gpudata, self.gpudata, result.gpudata, 
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

    def fill(self, value, stream=None):
        """fills the array with the specified value"""
        func = elementwise.get_fill_kernel(self.dtype)
        func.set_block_shape(*self._block)
        func.prepared_async_call(self._grid, stream,
                value, self.gpudata, self.mem_size)

        return self

    def bind_to_texref(self, texref, allow_offset=False):
        return texref.set_address(self.gpudata, self.nbytes, 
                allow_offset=allow_offset) / self.dtype.itemsize

    def bind_to_texref_ext(self, texref, channels=1, allow_double_hack=False, 
            allow_offset=False):
        if self.dtype == numpy.float64 and allow_double_hack:
            if channels != 1:
                raise ValueError, "'fake' double precision textures can only have one channel"

            channels = 2
            fmt = drv.array_format.SIGNED_INT32
            read_as_int = True
        else:
            fmt = drv.dtype_to_array_format(self.dtype)
            read_as_int = numpy.integer in self.dtype.type.__mro__

        offset = texref.set_address(self.gpudata, self.nbytes, allow_offset=allow_offset)
        texref.set_format(fmt, channels)

        if read_as_int:
            texref.set_flags(texref.get_flags() | drv.TRSF_READ_AS_INTEGER)

        return offset/self.dtype.itemsize

    def __len__(self):
        """Return the size of the leading dimension of self."""
        if len(self.shape):
            return self.shape[0]
        else:
            return 1

    def __abs__(self):
        """Return a `GPUArray` of the absolute values of the elements
        of `self`.
        """

        result = self._new_like_me()

        if self.dtype == numpy.float32:
            fname = "fabsf"
        elif self.dtype == numpy.float64:
            fname = "fabs"
        else:
            fname = "abs"

        func = elementwise.get_unary_func_kernel(fname, self.dtype)
        func.set_block_shape(*self._block)
        func.prepared_async_call(self._grid, None,
                self.gpudata,result.gpudata, self.mem_size)

        return result

    def __pow__(self, other):
        """pow function::
 
           example:
                   array = pow(array)
                   array = pow(array,4)
                   array = pow(array,array)

        """

        if isinstance(other, GPUArray):
            assert self.shape == other.shape

            result = self._new_like_me(_get_common_dtype(self, other))

            func = elementwise.get_pow_array_kernel(
                    self.dtype, other.dtype, result.dtype)

            func.set_block_shape(*self._block)
            func.prepared_async_call(self._grid, None,
                    self.gpudata, other.gpudata, result.gpudata,
                    self.mem_size)
            
            return result
        else:
            result = self._new_like_me()
            func = elementwise.get_pow_kernel(self.dtype)
            func.set_block_shape(*self._block)
            func.prepared_async_call(self._grid, None,
                    other, self.gpudata, result.gpudata,
                    self.mem_size)

            return result

    def reverse(self, stream=None):
        """Return this array in reversed order. The array is treated
        as one-dimensional.
        """

        result = self._new_like_me()

        func = elementwise.get_reverse_kernel(self.dtype)
        func.set_block_shape(*self._block)
        func.prepared_async_call(self._grid, stream,
                self.gpudata, result.gpudata,
                self.mem_size)

        return result

    def astype(self, dtype, stream=None):
        if dtype == self.dtype:
            return self

        result = self._new_like_me(dtype=dtype)

        func = elementwise.get_copy_kernel(dtype, self.dtype)
        func.set_block_shape(*self._block)
        func.prepared_async_call(self._grid, stream,
                result.gpudata, self.gpudata,
                self.mem_size)

        return result


    # slicing -----------------------------------------------------------------
    def __getitem__(self, idx):
        if idx == ():
            return self

        if len(self.shape) > 1:
            raise NotImplementedError("multi-d slicing is not yet implemented")

        if not isinstance(idx, slice):
            raise ValueError("non-slice indexing not supported: %s" % (idx,))

        l, = self.shape
        start, stop, stride = idx.indices(l)

        if stride != 1:
            raise NotImplementedError("strided slicing is not yet implemented")

        return GPUArray(
                shape=((stop-start)//stride,),
                dtype=self.dtype,
                allocator=self.allocator,
                base=self,
                gpudata=int(self.gpudata) + start*self.dtype.itemsize)

    # complex-valued business -------------------------------------------------
    @property
    def real(self):
        dtype = self.dtype
        if issubclass(dtype.type, numpy.complexfloating):
            from pytools import match_precision
            real_dtype = match_precision(numpy.dtype(numpy.float64), dtype)

            result = self._new_like_me(dtype=real_dtype)

            func = elementwise.get_real_kernel(dtype, real_dtype)
            func.set_block_shape(*self._block)
            func.prepared_async_call(self._grid, None,
                    self.gpudata, result.gpudata,
                    self.mem_size)

            return result
        else:
            return self

    @property
    def imag(self):
        dtype = self.dtype
        if issubclass(self.dtype.type, numpy.complexfloating):
            from pytools import match_precision
            real_dtype = match_precision(numpy.dtype(numpy.float64), dtype)

            result = self._new_like_me(dtype=real_dtype)

            func = elementwise.get_imag_kernel(dtype, real_dtype)
            func.set_block_shape(*self._block)
            func.prepared_async_call(self._grid, None,
                    self.gpudata, result.gpudata,
                    self.mem_size)

            return result
        else:
            return zeros_like(self)

    def conj(self):
        dtype = self.dtype
        if issubclass(self.dtype.type, numpy.complexfloating):
            result = self._new_like_me()

            func = elementwise.get_conj_kernel(dtype)
            func.set_block_shape(*self._block)
            func.prepared_async_call(self._grid, None,
                    self.gpudata, result.gpudata,
                    self.mem_size)

            return result
        else:
            return self

    # rich comparisons (or rather, lack thereof) ------------------------------
    def __eq__(self, other):
        raise NotImplementedError

    def __ne__(self, other):
        raise NotImplementedError

    def __le__(self, other):
        raise NotImplementedError

    def __ge__(self, other):
        raise NotImplementedError

    def __lt__(self, other):
        raise NotImplementedError

    def __gt__(self, other):
        raise NotImplementedError



def to_gpu(ary, allocator=drv.mem_alloc):
    """converts a numpy array to a GPUArray"""
    result = GPUArray(ary.shape, ary.dtype, allocator)
    result.set(ary)
    return result

def to_gpu_async(ary, allocator=drv.mem_alloc, stream=None):
    """converts a numpy array to a GPUArray"""
    result = GPUArray(ary.shape, ary.dtype, allocator)
    result.set_async(ary, stream)
    return result

empty = GPUArray

def zeros(shape, dtype, allocator=drv.mem_alloc):
    """Returns an array of the given shape and dtype filled with 0's."""

    result = GPUArray(shape, dtype, allocator)
    result.fill(0)
    return result

def empty_like(other_ary):
    result = GPUArray(
            other_ary.shape, other_ary.dtype, other_ary.allocator)
    return result

def zeros_like(other_ary):
    result = GPUArray(
            other_ary.shape, other_ary.dtype, other_ary.allocator)
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
    from pytools import Record
    class Info(Record):
        pass

    explicit_dtype = False

    inf = Info()
    inf.start = None
    inf.stop = None
    inf.step = None
    inf.dtype = None

    if isinstance(args[-1], numpy.dtype):
        dtype = args[-1]
        args = args[:-1]
        explicit_dtype = True

    argc = len(args)
    if argc == 0:
        raise ValueError, "stop argument required"
    elif argc == 1:
        inf.stop = args[0]
    elif argc == 2:
        inf.start = args[0]
        inf.stop = args[1]
    elif argc == 3:
        inf.start = args[0]
        inf.stop = args[1]
        inf.step = args[2]
    else:
        raise ValueError, "too many arguments"

    admissible_names = ["start", "stop", "step", "dtype"]
    for k, v in kwargs.iteritems():
        if k in admissible_names:
            if getattr(inf, k) is None:
                setattr(inf, k, v)
                if k == "dtype":
                    explicit_dtype = True
            else:
                raise ValueError, "may not specify '%s' by position and keyword" % k
        else:
            raise ValueError, "unexpected keyword argument '%s'" % k

    if inf.start is None:
        inf.start = 0
    if inf.step is None:
        inf.step = 1
    if inf.dtype is None:
        inf.dtype = numpy.array([inf.start, inf.stop, inf.step]).dtype

    # actual functionality ----------------------------------------------------
    dtype = numpy.dtype(inf.dtype)
    start = dtype.type(inf.start)
    step = dtype.type(inf.step)
    stop = dtype.type(inf.stop)

    if not explicit_dtype and dtype != numpy.float32:
        from warnings import warn
        warn("behavior change: arange guessed dtype other than float32. "
                "suggest specifying explicit dtype.")

    from math import ceil
    size = int(ceil((stop-start)/step))
  
    result = GPUArray((size,), dtype)

    func = elementwise.get_arange_kernel(dtype)
    func.set_block_shape(*result._block)
    func.prepared_async_call(result._grid, kwargs.get("stream"),
            result.gpudata, start, step, size)

    return result




def take(a, indices, out=None, stream=None):
    if out is None:
        out = GPUArray(indices.shape, a.dtype, a.allocator)

    assert len(indices.shape) == 1

    func, tex_src = elementwise.get_take_kernel(a.dtype, indices.dtype)
    a.bind_to_texref_ext(tex_src[0], allow_double_hack=True)

    func.set_block_shape(*out._block)
    func.prepared_async_call(out._grid, stream,
            indices.gpudata, out.gpudata, indices.size)

    return out




def multi_take(arrays, indices, out=None, stream=None):
    if not len(arrays):
        return []

    assert len(indices.shape) == 1

    from pytools import single_valued
    a_dtype = single_valued(a.dtype for a in arrays)
    a_allocator = arrays[0].dtype

    vec_count = len(arrays)

    if out is None:
        out = [GPUArray(indices.shape, a_dtype, a_allocator)
                for i in range(vec_count)]
    else:
        if len(out) != len(arrays):
            raise ValueError("out and arrays must have the same length")

    chunk_size = _builtin_min(vec_count, 20)

    def make_func_for_chunk_size(chunk_size):
        func, tex_src = elementwise.get_take_kernel(a_dtype, indices.dtype, 
                vec_count=chunk_size)
        func.set_block_shape(*indices._block)
        return func, tex_src

    func, tex_src = make_func_for_chunk_size(chunk_size)

    for start_i in range(0, len(arrays), chunk_size):
        chunk_slice = slice(start_i, start_i+chunk_size)

        if start_i + chunk_size > vec_count:
            func, tex_src = make_func_for_chunk_size(vec_count-start_i)

        for i, a in enumerate(arrays[chunk_slice]):
            a.bind_to_texref_ext(tex_src[i], allow_double_hack=True)

        func.prepared_async_call(indices._grid, stream, 
                indices.gpudata, 
                *([o.gpudata for o in out[chunk_slice]]
                    + [indices.size]))

    return out




def multi_take_put(arrays, dest_indices, src_indices, dest_shape=None, 
        out=None, stream=None, src_offsets=None):
    if not len(arrays):
        return []

    from pytools import single_valued
    a_dtype = single_valued(a.dtype for a in arrays)
    a_allocator = arrays[0].allocator

    vec_count = len(arrays)

    if out is None:
        out = [GPUArray(dest_shape, a_dtype, a_allocator)
                for i in range(vec_count)]
    else:
        if a_dtype != single_valued(o.dtype for o in out):
            raise TypeError("arrays and out must have the same dtype")
        if len(out) != vec_count:
            raise ValueError("out and arrays must have the same length")

    if src_indices.dtype != dest_indices.dtype:
        raise TypeError("src_indices and dest_indices must have the same dtype")

    if len(src_indices.shape) != 1:
        raise ValueError("src_indices must be 1D")

    if src_indices.shape != dest_indices.shape:
        raise ValueError("src_indices and dest_indices must have the same shape")

    if src_offsets is None:
        src_offsets_list = []
        max_chunk_size = 20
    else:
        src_offsets_list = src_offsets
        if len(src_offsets) != vec_count:
            raise ValueError("src_indices and src_offsets must have the same length")
        max_chunk_size = 10

    chunk_size = _builtin_min(vec_count, max_chunk_size)

    def make_func_for_chunk_size(chunk_size):
        func, tex_src = elementwise.get_take_put_kernel(
                a_dtype, src_indices.dtype, 
                with_offsets=src_offsets is not None,
                vec_count=chunk_size)

        func.set_block_shape(*src_indices._block)
        return func, tex_src

    func, tex_src = make_func_for_chunk_size(chunk_size)

    for start_i in range(0, len(arrays), chunk_size):
        chunk_slice = slice(start_i, start_i+chunk_size)

        if start_i + chunk_size > vec_count:
            func, tex_src = make_func_for_chunk_size(vec_count-start_i)

        for src_tr, a in zip(tex_src, arrays[chunk_slice]):
            a.bind_to_texref_ext(src_tr, allow_double_hack=True)

        func.prepared_async_call(src_indices._grid, stream,
                dest_indices.gpudata, src_indices.gpudata,
                *([o.gpudata for o in out[chunk_slice]]
                    + src_offsets_list[chunk_slice]
                    + [src_indices.size]))

    return out




def multi_put(arrays, dest_indices, dest_shape=None, out=None, stream=None):
    if not len(arrays):
        return []

    from pytools import single_valued
    a_dtype = single_valued(a.dtype for a in arrays)
    a_allocator = arrays[0].allocator

    vec_count = len(arrays)

    if out is None:
        out = [GPUArray(dest_shape, a_dtype, a_allocator)
                for i in range(vec_count)]
    else:
        if a_dtype != single_valued(o.dtype for o in out):
            raise TypeError("arrays and out must have the same dtype")
        if len(out) != vec_count:
            raise ValueError("out and arrays must have the same length")

    if len(dest_indices.shape) != 1:
        raise ValueError("src_indices must be 1D")

    chunk_size = _builtin_min(vec_count, 10)

    def make_func_for_chunk_size(chunk_size):
        func = elementwise.get_put_kernel(
                a_dtype, dest_indices.dtype, vec_count=chunk_size)
        func.set_block_shape(*dest_indices._block)
        return func

    func = make_func_for_chunk_size(chunk_size)

    for start_i in range(0, len(arrays), chunk_size):
        chunk_slice = slice(start_i, start_i+chunk_size)

        if start_i + chunk_size > vec_count:
            func = make_func_for_chunk_size(vec_count-start_i)

        func.prepared_async_call(dest_indices._grid, stream,
                dest_indices.gpudata, 
                *([o.gpudata for o in out[chunk_slice]]
                    + [i.gpudata for i in arrays[chunk_slice]]
                    + [dest_indices.size]))

    return out




def if_positive(criterion, then_, else_, out=None, stream=None):
    if not (criterion.shape == then_.shape == else_.shape):
        raise ValueError("shapes do not match")

    if not (then_.dtype == else_.dtype):
        raise ValueError("dtypes do not match")

    func = elementwise.get_if_positive_kernel(
            criterion.dtype, then_.dtype)

    if out is None:
        out = empty_like(then_)

    func.set_block_shape(*criterion._block)
    func.prepared_async_call(criterion._grid, stream,
            criterion.gpudata, then_.gpudata, else_.gpudata, out.gpudata,
            criterion.size)

    return out




def maximum(a, b, out=None, stream=None):
    # silly, but functional
    return if_positive(a.mul_add(1, b, -1, stream=stream), a, b, 
            stream=stream, out=out)




def minimum(a, b, out=None, stream=None):
    # silly, but functional
    return if_positive(a.mul_add(1, b, -1, stream=stream), b, a, 
            stream=stream, out=out)




# reductions ------------------------------------------------------------------
def sum(a, dtype=None, stream=None):
    from pycuda.reduction import get_sum_kernel
    krnl = get_sum_kernel(dtype, a.dtype)
    return krnl(a, stream=stream)

def dot(a, b, dtype=None, stream=None):
    from pycuda.reduction import get_dot_kernel
    krnl = get_dot_kernel(dtype, a.dtype, b.dtype)
    return krnl(a, b, stream=stream)

def subset_dot(subset, a, b, dtype=None, stream=None):
    from pycuda.reduction import get_subset_dot_kernel
    krnl = get_subset_dot_kernel(dtype, a.dtype, b.dtype)
    return krnl(subset, a, b, stream=stream)

def _make_minmax_kernel(what):
    def f(a, stream=None):
        from pycuda.reduction import get_minmax_kernel
        krnl = get_minmax_kernel(what, a.dtype)
        return krnl(a,  stream=stream)

    return f

_builtin_min = min
_builtin_max = max
min = _make_minmax_kernel("min")
max = _make_minmax_kernel("max")

def _make_subset_minmax_kernel(what):
    def f(subset, a, stream=None):
        from pycuda.reduction import get_subset_minmax_kernel
        import pycuda.reduction
        krnl = get_subset_minmax_kernel(what, a.dtype)
        return krnl(subset, a,  stream=stream)

    return f

subset_min = _make_subset_minmax_kernel("min")
subset_max = _make_subset_minmax_kernel("max")
