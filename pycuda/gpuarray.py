__copyright__ = """
Copyright 2008-2021 Andreas Kloeckner
Copyright 2021 NVIDIA Corporation
"""

import numpy as np
import pycuda.elementwise as elementwise
from pytools import memoize, memoize_method
import pycuda.driver as drv
from pycuda.compyte.array import (
    as_strided as _as_strided,
    f_contiguous_strides as _f_contiguous_strides,
    c_contiguous_strides as _c_contiguous_strides,
    ArrayFlags as _ArrayFlags,
    get_common_dtype as _get_common_dtype_base,
)
from pycuda.characterize import has_double_support
from functools import reduce
import numbers

import copyreg


def _get_common_dtype(obj1, obj2):
    return _get_common_dtype_base(obj1, obj2, has_double_support())


def _get_broadcasted_binary_op_result(obj1, obj2,
                                      dtype_getter=_get_common_dtype):

    if obj1.shape == obj2.shape:
        return obj1._new_like_me(dtype_getter(obj1, obj2))
    elif obj1.shape == ():
        return obj2._new_like_me(dtype_getter(obj1, obj2))
    elif obj2.shape == ():
        return obj1._new_like_me(dtype_getter(obj1, obj2))
    else:
        raise NotImplementedError("Broadcasting binary operator with shapes:"
                                  f" {obj1.shape}, {obj2.shape}.")
# {{{ vector types


class vec:  # noqa
    pass


def _create_vector_types():
    from pycuda.characterize import platform_bits

    if platform_bits() == 32:
        long_dtype = np.int32
        ulong_dtype = np.uint32
    else:
        long_dtype = np.int64
        ulong_dtype = np.uint64

    field_names = ["x", "y", "z", "w"]

    from pycuda.tools import get_or_register_dtype

    for base_name, base_type, counts in [
        ("char", np.int8, [1, 2, 3, 4]),
        ("uchar", np.uint8, [1, 2, 3, 4]),
        ("short", np.int16, [1, 2, 3, 4]),
        ("ushort", np.uint16, [1, 2, 3, 4]),
        ("int", np.int32, [1, 2, 3, 4]),
        ("uint", np.uint32, [1, 2, 3, 4]),
        ("long", long_dtype, [1, 2, 3, 4]),
        ("ulong", ulong_dtype, [1, 2, 3, 4]),
        ("longlong", np.int64, [1, 2]),
        ("ulonglong", np.uint64, [1, 2]),
        ("float", np.float32, [1, 2, 3, 4]),
        ("double", np.float64, [1, 2]),
    ]:
        for count in counts:
            name = "%s%d" % (base_name, count)
            dtype = np.dtype([(field_names[i], base_type) for i in range(count)])

            get_or_register_dtype(name, dtype)

            setattr(vec, name, dtype)

            my_field_names = ",".join(field_names[:count])
            setattr(
                vec,
                "make_" + name,
                staticmethod(
                    eval(
                        "lambda %s: array((%s), dtype=my_dtype)"
                        % (my_field_names, my_field_names),
                        {"array": np.array, "my_dtype": dtype},
                    )
                ),
            )


_create_vector_types()

# }}}


# {{{ helper functionality


@memoize
def _splay_backend(n, dev):
    # heavily modified from cublas
    from pycuda.tools import DeviceData

    devdata = DeviceData(dev)

    min_threads = devdata.warp_size
    max_threads = 128
    max_blocks = (
        4
        * devdata.thread_blocks_per_mp
        * dev.get_attribute(drv.device_attribute.MULTIPROCESSOR_COUNT)
    )

    if n < min_threads:
        block_count = 1
        threads_per_block = min_threads
    elif n < (max_blocks * min_threads):
        block_count = (n + min_threads - 1) // min_threads
        threads_per_block = min_threads
    elif n < (max_blocks * max_threads):
        block_count = max_blocks
        grp = (n + min_threads - 1) // min_threads
        threads_per_block = ((grp + max_blocks - 1) // max_blocks) * min_threads
    else:
        block_count = max_blocks
        threads_per_block = max_threads

    # print "n:%d bc:%d tpb:%d" % (n, block_count, threads_per_block)
    return (block_count, 1), (threads_per_block, 1, 1)


def splay(n, dev=None):
    if dev is None:
        dev = drv.Context.get_device()
    return _splay_backend(n, dev)


# }}}


# {{{ main GPUArray class


def _make_binary_op(operator):
    def func(self, other):
        if not self.flags.forc:
            raise RuntimeError(
                "only contiguous arrays may " "be used as arguments to this operation"
            )
        if isinstance(other, GPUArray) and (self, GPUArray):
            if not other.flags.forc:
                raise RuntimeError(
                    "only contiguous arrays may "
                    "be used as arguments to this operation"
                )

            result = _get_broadcasted_binary_op_result(self, other)
            func = elementwise.get_binary_op_kernel(
                self.dtype,
                other.dtype,
                result.dtype,
                operator,
                x_is_scalar=(self.shape == ()),
                y_is_scalar=(other.shape == ()))

            func.prepared_async_call(
                self._grid,
                self._block,
                None,
                self.gpudata,
                other.gpudata,
                result.gpudata,
                self.mem_size,
            )

            return result
        elif isinstance(self, GPUArray):  # scalar operator
            assert np.isscalar(other)
            result = self._new_like_me()
            func = elementwise.get_scalar_op_kernel(self.dtype,
                                                    np.dtype(type(other)),
                                                    result.dtype, operator)
            func.prepared_async_call(
                self._grid,
                self._block,
                None,
                self.gpudata,
                other,
                result.gpudata,
                self.mem_size,
            )
            return result
        else:
            return AssertionError

    return func


class GPUArray:
    """A GPUArray is used to do array-based calculation on the GPU.

    This is mostly supposed to be a numpy-workalike. Operators
    work on an element-by-element basis, just like numpy.ndarray.
    """

    __array_priority__ = 100

    def __init__(
        self,
        shape,
        dtype,
        allocator=drv.mem_alloc,
        base=None,
        gpudata=None,
        strides=None,
        order="C",
    ):
        dtype = np.dtype(dtype)

        try:
            s = 1
            for dim in shape:
                s *= dim
        except TypeError:
            # handle dim-0 ndarrays:
            if isinstance(shape, np.ndarray):
                shape = shape.item()
            assert isinstance(shape, numbers.Integral)
            s = shape
            shape = (shape,)
        else:
            # handle shapes that are ndarrays
            shape = tuple(shape)

        if isinstance(s, np.integer):
            # bombs if s is a Python integer
            s = s.item()

        # Make sure shape is made of int and not e.g. np.int32 as these can overflow
        # e.g. in __getitem__() when adding the new_offset...
        shape = tuple(int(v) for v in shape)

        if strides is None:
            if order == "F":
                strides = _f_contiguous_strides(dtype.itemsize, shape)
            elif order == "C":
                strides = _c_contiguous_strides(dtype.itemsize, shape)
            else:
                raise ValueError("invalid order: %s" % order)
        else:
            # FIXME: We should possibly perform some plausibility
            # checking on 'strides' here.

            strides = tuple(strides)

        strides = tuple(int(v) for v in strides)

        self.shape = shape
        self.dtype = dtype
        self.strides = strides
        self.mem_size = self.size = s
        self.nbytes = self.dtype.itemsize * self.size
        self.itemsize = self.dtype.itemsize

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

    @property
    def __cuda_array_interface__(self):
        """Returns a CUDA Array Interface dictionary describing this array's
        data."""
        if self.gpudata is not None:
            ptr = int(self.gpudata)
        else:
            ptr = 0

        return {
            "shape": self.shape,
            "strides": self.strides,
            # data is a tuple: (ptr, readonly) - always export GPUArray
            # instances as read-write
            "data": (ptr, False),
            "typestr": self.dtype.str,
            "stream": None,
            "version": 3
        }

    @property
    def ndim(self):
        return len(self.shape)

    @property
    @memoize_method
    def flags(self):
        return _ArrayFlags(self)

    def set(self, ary, async_=False, stream=None, **kwargs):
        # {{{ handle 'async' deprecation

        async_arg = kwargs.pop("async", None)
        if async_arg is not None:
            if async_ is not None:
                raise TypeError("may not specify both 'async' and 'async_'")
            async_ = async_arg

        if async_ is None:
            async_ = False

        if kwargs:
            raise TypeError("extra keyword arguments specified: %s" % ", ".join(kwargs))

        # }}}

        if ary.size != self.size:
            raise ValueError("ary and self must be the same size")
        if ary.shape != self.shape:
            from warnings import warn

            warn("Setting array from one with different shape.", stacklevel=2)
            ary = ary.reshape(self.shape)

        if ary.dtype != self.dtype:
            raise ValueError("ary and self must have the same dtype")

        if self.size:
            _memcpy_discontig(self, ary, async_=async_, stream=stream)

    def set_async(self, ary, stream=None):
        return self.set(ary, async_=True, stream=stream)

    def get(self, ary=None, pagelocked=False, async_=False, stream=None, **kwargs):
        # {{{ handle 'async' deprecation

        async_arg = kwargs.pop("async", None)
        if async_arg is not None:
            if async_ is not None:
                raise TypeError("may not specify both 'async' and 'async_'")
            async_ = async_arg

        if async_ is None:
            async_ = False

        if kwargs:
            raise TypeError("extra keyword arguments specified: %s" % ", ".join(kwargs))

        # }}}

        if ary is None:
            if pagelocked:
                ary = drv.pagelocked_empty(self.shape, self.dtype)
            else:
                ary = np.empty(self.shape, self.dtype)

            strides = _compact_strides(self)
            ary = _as_strided(ary, strides=strides)
        else:
            if self.size != ary.size:
                raise ValueError("self and ary must be the same size")
            if self.shape != ary.shape:
                from warnings import warn

                warn(
                    "get() between arrays of different shape is deprecated "
                    "and will be removed in PyCUDA 2017.x",
                    DeprecationWarning,
                    stacklevel=2,
                )
                ary = ary.reshape(self.shape)

            if self.dtype != ary.dtype:
                raise TypeError("self and ary must have the same dtype")

        if self.size:
            _memcpy_discontig(ary, self, async_=async_, stream=stream)
        return ary

    def get_async(self, stream=None, ary=None):
        return self.get(ary=ary, async_=True, stream=stream)

    def copy(self):
        new = GPUArray(self.shape, self.dtype, self.allocator, strides=self.strides)
        _memcpy_discontig(new, self)
        return new

    def __str__(self):
        return str(self.get())

    def __repr__(self):
        return repr(self.get())

    def __hash__(self):
        raise TypeError("GPUArrays are not hashable.")

    def __bool__(self):
        if self.size == 1:
            return bool(self.get())
        else:
            raise ValueError("The truth value of an array with "
                    "more than one element is ambiguous. Use a.any() or a.all()")

    @property
    def ptr(self):
        return self.gpudata.__int__()

    # kernel invocation wrappers ----------------------------------------------
    def _axpbyz(self, selffac, other, otherfac, out, add_timer=None, stream=None):
        """Compute ``out = selffac * self + otherfac*other``,
        where `other` is a vector.."""
        if not self.flags.forc or not other.flags.forc:
            raise RuntimeError(
                "only contiguous arrays may " "be used as arguments to this operation"
            )
        assert ((self.shape == other.shape == out.shape)
            or ((self.shape == ()) and other.shape == out.shape)
            or ((other.shape == ()) and self.shape == out.shape))
        func = elementwise.get_axpbyz_kernel(
            self.dtype, other.dtype, out.dtype,
            x_is_scalar=(self.shape == ()),
            y_is_scalar=(other.shape == ()))
        if add_timer is not None:
            add_timer(
                3 * self.size,
                func.prepared_timed_call(
                    out._grid,
                    selffac,
                    out.gpudata,
                    otherfac,
                    other.gpudata,
                    out.gpudata,
                    out.mem_size,
                ),
            )
        else:
            func.prepared_async_call(
                out._grid,
                out._block,
                stream,
                selffac,
                self.gpudata,
                otherfac,
                other.gpudata,
                out.gpudata,
                out.mem_size,
            )

        return out

    def _axpbz(self, selffac, other, out, stream=None):
        """Compute ``out = selffac * self + other``, where `other` is a scalar."""

        if not self.flags.forc:
            raise RuntimeError(
                "only contiguous arrays may " "be used as arguments to this operation"
            )

        func = elementwise.get_axpbz_kernel(self.dtype, out.dtype)
        func.prepared_async_call(
            self._grid,
            self._block,
            stream,
            selffac,
            self.gpudata,
            other,
            out.gpudata,
            self.mem_size,
        )

        return out

    def _elwise_multiply(self, other, out, stream=None):
        if not self.flags.forc:
            raise RuntimeError(
                "only contiguous arrays may " "be used as arguments to this operation"
            )
        assert ((self.shape == other.shape == out.shape)
            or ((self.shape == ()) and other.shape == out.shape)
            or ((other.shape == ()) and self.shape == out.shape))

        func = elementwise.get_binary_op_kernel(
            self.dtype,
            other.dtype,
            out.dtype,
            "*",
            x_is_scalar=(self.shape == ()),
            y_is_scalar=(other.shape == ()))

        func.prepared_async_call(
            out._grid,
            out._block,
            stream,
            self.gpudata,
            other.gpudata,
            out.gpudata,
            out.mem_size,
        )

        return out

    def _rdiv_scalar(self, other, out, stream=None):
        """Divides an array by a scalar::

        y = n / self
        """

        if not self.flags.forc:
            raise RuntimeError(
                "only contiguous arrays may " "be used as arguments to this operation"
            )

        func = elementwise.get_rdivide_elwise_kernel(self.dtype, out.dtype)
        func.prepared_async_call(
            self._grid,
            self._block,
            stream,
            self.gpudata,
            other,
            out.gpudata,
            self.mem_size,
        )

        return out

    def _div(self, other, out, stream=None):
        """Divides an array by another array."""

        if not self.flags.forc or not other.flags.forc:
            raise RuntimeError(
                "only contiguous arrays may " "be used as arguments to this operation"
            )

        assert ((self.shape == other.shape == out.shape)
            or ((self.shape == ()) and other.shape == out.shape)
            or ((other.shape == ()) and self.shape == out.shape))

        func = elementwise.get_binary_op_kernel(
            self.dtype,
            other.dtype,
            out.dtype,
            "/",
            x_is_scalar=(self.shape == ()),
            y_is_scalar=(other.shape == ()))
        func.prepared_async_call(
            out._grid,
            out._block,
            stream,
            self.gpudata,
            other.gpudata,
            out.gpudata,
            out.mem_size,
        )

        return out

    def _new_like_me(self, dtype=None, order="C"):
        strides = None
        if dtype is None:
            dtype = self.dtype
        if dtype == self.dtype:
            strides = self.strides

        return self.__class__(
            self.shape, dtype, allocator=self.allocator, strides=strides, order=order
        )

    # operators ---------------------------------------------------------------
    def mul_add(self, selffac, other, otherfac, add_timer=None, stream=None, out=None):
        """Return `selffac * self + otherfac*other`."""
        result = out if out is not None else self._new_like_me(_get_common_dtype(self, other))
        return self._axpbyz(selffac, other, otherfac, result, add_timer, stream=stream)

    def __add__(self, other):
        """Add an array with an array or an array with a scalar."""

        if isinstance(other, GPUArray):
            # add another vector
            result = _get_broadcasted_binary_op_result(self, other)
            return self._axpbyz(1, other, 1, result)

        elif np.isscalar(other):
            # add a scalar
            if other == 0:
                return self.copy()
            else:
                result = self._new_like_me(_get_common_dtype(self, other))
                return self._axpbz(1, other, result)
        else:
            return NotImplemented
    __radd__ = __add__

    def __sub__(self, other):
        """Substract an array from an array or a scalar from an array."""

        if isinstance(other, GPUArray):
            result = _get_broadcasted_binary_op_result(self, other)
            return self._axpbyz(1, other, -1, result)
        elif np.isscalar(other):
            if other == 0:
                return self.copy()
            else:
                # create a new array for the result
                result = self._new_like_me(_get_common_dtype(self, other))
                return self._axpbz(1, -other, result)
        else:
            return NotImplemented

    def __rsub__(self, other):
        """Substracts an array by a scalar or an array::

        x = n - self
        """
        # other must be a scalar
        result = self._new_like_me(_get_common_dtype(self, other))
        return self._axpbz(-1, other, result)

    def __iadd__(self, other):
        if isinstance(other, GPUArray):
            return self._axpbyz(1, other, 1, self)
        else:
            return self._axpbz(1, other, self)

    def __isub__(self, other):
        if isinstance(other, GPUArray):
            return self._axpbyz(1, other, -1, self)
        else:
            return self._axpbz(1, -other, self)

    def __pos__(self):
        return self

    def __neg__(self):
        result = self._new_like_me()
        return self._axpbz(-1, 0, result)

    def __mul__(self, other):
        if isinstance(other, GPUArray):
            result = _get_broadcasted_binary_op_result(self, other)
            return self._elwise_multiply(other, result)
        elif np.isscalar(other):
            result = self._new_like_me(_get_common_dtype(self, other))
            return self._axpbz(other, 0, result)
        else:
            return NotImplemented

    def __rmul__(self, scalar):
        result = self._new_like_me(_get_common_dtype(self, scalar))
        return self._axpbz(scalar, 0, result)

    def __imul__(self, other):
        if isinstance(other, GPUArray):
            return self._elwise_multiply(other, self)
        else:
            return self._axpbz(other, 0, self)

    def __div__(self, other):
        """Divides an array by an array or a scalar::

        x = self / n
        """
        if isinstance(other, GPUArray):
            result = _get_broadcasted_binary_op_result(self, other)
            return self._div(other, result)
        elif np.isscalar(other):
            if other == 1:
                return self.copy()
            else:
                # create a new array for the result
                result = self._new_like_me(_get_common_dtype(self, other))
                return self._axpbz(1 / other, 0, result)
        else:
            return NotImplemented
    __truediv__ = __div__

    def __rdiv__(self, other):
        """Divides an array by a scalar or an array::

        x = n / self
        """
        # create a new array for the result
        result = self._new_like_me(_get_common_dtype(self, other))
        return self._rdiv_scalar(other, result)

    __rtruediv__ = __rdiv__

    def __idiv__(self, other):
        """Divides an array by an array or a scalar::

        x /= n
        """
        if isinstance(other, GPUArray):
            return self._div(other, self)
        else:
            if other == 1:
                return self
            else:
                return self._axpbz(1 / other, 0, self)

    __itruediv__ = __idiv__

    def fill(self, value, stream=None):
        """fills the array with the specified value"""
        if not self.flags.forc:
            raise RuntimeError(
                "only contiguous arrays may be used as arguments to this operation")

        func = elementwise.get_fill_kernel(self.dtype)
        func.prepared_async_call(
            self._grid, self._block, stream, value, self.gpudata, self.mem_size
        )

        return self

    def bind_to_texref(self, texref, allow_offset=False):
        return (
            texref.set_address(self.gpudata, self.nbytes, allow_offset=allow_offset)
            / self.dtype.itemsize
        )

    def bind_to_texref_ext(
        self,
        texref,
        channels=1,
        allow_double_hack=False,
        allow_complex_hack=False,
        allow_offset=False,
    ):
        if not self.flags.forc:
            raise RuntimeError(
                "only contiguous arrays may be used as arguments to this operation")

        if self.dtype == np.float64 and allow_double_hack:
            if channels != 1:
                raise ValueError(
                    "'fake' double precision textures can " "only have one channel"
                )

            channels = 2
            fmt = drv.array_format.SIGNED_INT32
            read_as_int = True
        elif self.dtype == np.complex64 and allow_complex_hack:
            if channels != 1:
                raise ValueError(
                    "'fake' complex64 textures can " "only have one channel"
                )

            channels = 2
            fmt = drv.array_format.UNSIGNED_INT32
            read_as_int = True
        elif self.dtype == np.complex128 and allow_complex_hack:
            if channels != 1:
                raise ValueError(
                    "'fake' complex128 textures can " "only have one channel"
                )

            channels = 4
            fmt = drv.array_format.SIGNED_INT32
            read_as_int = True
        else:
            fmt = drv.dtype_to_array_format(self.dtype)
            read_as_int = np.integer in self.dtype.type.__mro__

        offset = texref.set_address(
            self.gpudata, self.nbytes, allow_offset=allow_offset
        )
        texref.set_format(fmt, channels)

        if read_as_int:
            texref.set_flags(texref.get_flags() | drv.TRSF_READ_AS_INTEGER)

        return offset / self.dtype.itemsize

    def __len__(self):
        """Return the size of the leading dimension of self."""
        if len(self.shape):
            return self.shape[0]
        else:
            raise TypeError("len() of unsized object")

    def __abs__(self):
        """Return a `GPUArray` of the absolute values of the elements
        of `self`.
        """

        result = self._new_like_me()

        if self.dtype == np.float32:
            fname = "fabsf"
        elif self.dtype == np.float64:
            fname = "fabs"
        else:
            fname = "abs"

        if issubclass(self.dtype.type, np.complexfloating):
            from pytools import match_precision

            out_dtype = match_precision(np.dtype(np.float64), self.dtype)
            result = self._new_like_me(out_dtype)
        else:
            out_dtype = self.dtype

        func = elementwise.get_unary_func_kernel(fname, self.dtype, out_dtype=out_dtype)

        func.prepared_async_call(
            self._grid, self._block, None, self.gpudata, result.gpudata, self.mem_size
        )

        return result

    def _pow(self, other, new):
        """
        Do the pow operator.
        with new, the user can choose between ipow or just pow
        """
        common_dtype = _get_common_dtype(self, other)
        if new:
            result = self._new_like_me(common_dtype)
        else:
            result = self

        # {{{ sanity checks

        if (not self.flags.forc) or (isinstance(other, GPUArray)
                                     and not other.flags.forc):
            raise RuntimeError("only contiguous arrays may "
                               "be used as arguments to this operation")
        assert not isinstance(other, GPUArray) or other.shape == self.shape

        # }}}

        func = elementwise.get_pow_array_kernel(
            self.dtype,
            common_dtype if np.isscalar(other) else other.dtype,
            result.dtype,
            not np.isscalar(self),
            not np.isscalar(other)
        )

        func.prepared_async_call(
            self._grid,
            self._block,
            None,
            self.gpudata,
            other if np.isscalar(other) else other.gpudata,
            result.gpudata,
            self.mem_size,
        )

        return result

    def __pow__(self, other):
        """pow function::

        example:
                array = pow(array)
                array = pow(array,4)
                array = pow(array,array)

        """
        return self._pow(other, new=True)

    def __ipow__(self, other):
        """ipow function::

        example:
                array **= 4
                array **= array

        """
        return self._pow(other, new=False)

    def __rpow__(self, other):
        common_dtype = _get_common_dtype(self, other)
        result = self._new_like_me(common_dtype)

        if not np.isscalar(other):
            # Base is a gpuarray => do not cast.
            base = other
        else:
            base = common_dtype.type(other)

        func = elementwise.get_pow_array_kernel(
            base.dtype, self.dtype, result.dtype,
            is_base_array=not np.isscalar(other), is_exp_array=not np.isscalar(self))
        # Evaluates z = x ** y
        func.prepared_async_call(self._grid, self._block, None,
                                 base if np.isscalar(base) else base.gpudata,  # x
                                 self.gpudata,  # y
                                 result.gpudata,  # z
                                 self.mem_size)
        return result

    def reverse(self, stream=None):
        """Return this array in reversed order. The array is treated
        as one-dimensional.
        """

        if not self.flags.forc:
            raise RuntimeError(
                "only contiguous arrays may " "be used as arguments to this operation"
            )

        result = self._new_like_me()

        func = elementwise.get_reverse_kernel(self.dtype)
        func.prepared_async_call(
            self._grid, self._block, stream, self.gpudata, result.gpudata, self.mem_size
        )

        return result

    def astype(self, dtype, stream=None):
        if not self.flags.forc:
            raise RuntimeError(
                "only contiguous arrays may " "be used as arguments to this operation"
            )

        if dtype == self.dtype:
            return self.copy()

        result = self._new_like_me(dtype=dtype)

        func = elementwise.get_copy_kernel(dtype, self.dtype)
        func.prepared_async_call(
            self._grid, self._block, stream, result.gpudata, self.gpudata, self.mem_size
        )

        return result

    def any(self, stream=None, allocator=None):
        return any(self, stream=stream, allocator=allocator)

    def all(self, stream=None, allocator=None):
        return all(self, stream=stream, allocator=allocator)

    def reshape(self, *shape, **kwargs):
        """Gives a new shape to an array without changing its data."""

        # Python 2.x compatibility: use kwargs instead of named 'order' keyword
        order = kwargs.pop("order", "C")

        # TODO: add more error-checking, perhaps
        if not self.flags.forc:
            raise RuntimeError(
                "only contiguous arrays may " "be used as arguments to this operation"
            )

        if isinstance(shape[0], tuple) or isinstance(shape[0], list):
            shape = tuple(shape[0])

        same_contiguity = (order == "C" and self.flags.c_contiguous) or (
            order == "F" and self.flags.f_contiguous
        )

        if shape == self.shape and same_contiguity:
            return self

        if -1 in shape:
            shape = list(shape)
            idx = shape.index(-1)
            size = -reduce(lambda x, y: x * y, shape, 1)
            shape[idx] = self.size // size
            if -1 in shape[idx:]:
                raise ValueError("can only specify one unknown dimension")
            shape = tuple(shape)

        size = reduce(lambda x, y: x * y, shape, 1)
        if size != self.size:
            raise ValueError("total size of new array must be unchanged")

        return GPUArray(
            shape=shape,
            dtype=self.dtype,
            allocator=self.allocator,
            base=self,
            gpudata=int(self.gpudata),
            order=order,
        )

    def ravel(self):
        return self.reshape(self.size)

    def view(self, dtype=None):
        if dtype is None:
            dtype = self.dtype

        old_itemsize = self.dtype.itemsize
        itemsize = np.dtype(dtype).itemsize

        from pytools import argmin2

        min_stride_axis = argmin2(
            (axis, abs(stride)) for axis, stride in enumerate(self.strides)
        )

        if self.shape[min_stride_axis] * old_itemsize % itemsize != 0:
            raise ValueError("new type not compatible with array")

        new_shape = (
            self.shape[:min_stride_axis]
            + (self.shape[min_stride_axis] * old_itemsize // itemsize,)
            + self.shape[min_stride_axis + 1:]
        )
        new_strides = (
            self.strides[:min_stride_axis]
            + (self.strides[min_stride_axis] * itemsize // old_itemsize,)
            + self.strides[min_stride_axis + 1:]
        )

        return GPUArray(
            shape=new_shape,
            dtype=dtype,
            allocator=self.allocator,
            strides=new_strides,
            base=self,
            gpudata=int(self.gpudata),
        )

    def squeeze(self):
        """
        Returns a view of the array with dimensions of
        length 1 removed.
        """
        new_shape = tuple([dim for dim in self.shape if dim > 1])
        new_strides = tuple(
            [self.strides[i] for i, dim in enumerate(self.shape) if dim > 1]
        )

        return GPUArray(
            shape=new_shape,
            dtype=self.dtype,
            allocator=self.allocator,
            strides=new_strides,
            base=self,
            gpudata=int(self.gpudata),
        )

    def transpose(self, axes=None):
        """Permute the dimensions of an array.

        :arg axes: list of ints, optional.
            By default, reverse the dimensions, otherwise permute the axes
            according to the values given.

        :returns: :class:`GPUArray` A view of the array with its axes permuted.

        .. versionadded:: 2015.2
        """

        if axes is None:
            axes = range(self.ndim - 1, -1, -1)
        if len(axes) != len(self.shape):
            raise ValueError("axes don't match array")
        new_shape = [self.shape[axes[i]] for i in range(len(axes))]
        new_strides = [self.strides[axes[i]] for i in range(len(axes))]
        return GPUArray(
            shape=tuple(new_shape),
            dtype=self.dtype,
            allocator=self.allocator,
            base=self if self.base is None else self.base,
            gpudata=self.gpudata,
            strides=tuple(new_strides),
        )

    @property
    def T(self):  # noqa
        """
        .. versionadded:: 2015.2
        """
        return self.transpose()

    # {{{ slicing

    def __getitem__(self, index):
        """
        .. versionadded:: 2013.1
        """
        if not isinstance(index, tuple):
            index = (index,)

        new_shape = []
        new_offset = 0
        new_strides = []

        seen_ellipsis = False

        index_axis = 0
        array_axis = 0
        while index_axis < len(index):
            index_entry = index[index_axis]

            if array_axis > len(self.shape):
                raise IndexError("too many axes in index")

            if isinstance(index_entry, slice):
                start, stop, idx_stride = index_entry.indices(self.shape[array_axis])

                array_stride = self.strides[array_axis]

                new_shape.append((abs(stop - start) - 1) // abs(idx_stride) + 1)
                new_strides.append(idx_stride * array_stride)
                new_offset += array_stride * start

                index_axis += 1
                array_axis += 1

            elif isinstance(index_entry, (int, np.integer)):
                array_shape = self.shape[array_axis]
                if index_entry < 0:
                    index_entry += array_shape

                if not (0 <= index_entry < array_shape):
                    raise IndexError("subindex in axis %d out of range" % index_axis)

                new_offset += self.strides[array_axis] * index_entry

                index_axis += 1
                array_axis += 1

            elif index_entry is Ellipsis:
                index_axis += 1

                remaining_index_count = len(index) - index_axis
                new_array_axis = len(self.shape) - remaining_index_count
                if new_array_axis < array_axis:
                    raise IndexError("invalid use of ellipsis in index")
                while array_axis < new_array_axis:
                    new_shape.append(self.shape[array_axis])
                    new_strides.append(self.strides[array_axis])
                    array_axis += 1

                if seen_ellipsis:
                    raise IndexError("more than one ellipsis not allowed in index")
                seen_ellipsis = True

            elif index_entry is np.newaxis:
                new_shape.append(1)
                new_strides.append(0)
                index_axis += 1

            else:
                raise IndexError("invalid subindex in axis %d" % index_axis)

        while array_axis < len(self.shape):
            new_shape.append(self.shape[array_axis])
            new_strides.append(self.strides[array_axis])

            array_axis += 1

        return GPUArray(
            shape=tuple(new_shape),
            dtype=self.dtype,
            allocator=self.allocator,
            base=self,
            gpudata=int(self.gpudata) + new_offset,
            strides=tuple(new_strides),
        )

    def __setitem__(self, index, value):
        if np.isscalar(value):
            self[index].fill(value)
        else:
            _memcpy_discontig(self[index], value)

    # }}}

    # {{{ complex-valued business

    @property
    def real(self):
        dtype = self.dtype
        if issubclass(dtype.type, np.complexfloating):
            from pytools import match_precision

            real_dtype = match_precision(np.dtype(np.float64), dtype)
            if self.flags.f_contiguous:
                order = "F"
            else:
                order = "C"
            result = self._new_like_me(dtype=real_dtype, order=order)

            func = elementwise.get_real_kernel(dtype, real_dtype)
            func.prepared_async_call(
                self._grid,
                self._block,
                None,
                self.gpudata,
                result.gpudata,
                self.mem_size,
            )

            return result
        else:
            return self

    @property
    def imag(self):
        dtype = self.dtype
        if issubclass(self.dtype.type, np.complexfloating):
            if not self.flags.forc:
                raise RuntimeError(
                    "only contiguous arrays may "
                    "be used as arguments to this operation"
                )

            from pytools import match_precision

            real_dtype = match_precision(np.dtype(np.float64), dtype)
            if self.flags.f_contiguous:
                order = "F"
            else:
                order = "C"
            result = self._new_like_me(dtype=real_dtype, order=order)

            func = elementwise.get_imag_kernel(dtype, real_dtype)
            func.prepared_async_call(
                self._grid,
                self._block,
                None,
                self.gpudata,
                result.gpudata,
                self.mem_size,
            )

            return result
        else:
            return zeros_like(self)

    def conj(self, out=None):
        dtype = self.dtype
        if issubclass(self.dtype.type, np.complexfloating):
            if not self.flags.forc:
                raise RuntimeError(
                    "only contiguous arrays may "
                    "be used as arguments to this operation"
                )

            if self.flags.f_contiguous:
                order = "F"
            else:
                order = "C"
            if out is None:
                result = self._new_like_me(order=order)
            else:
                result = out

            func = elementwise.get_conj_kernel(dtype, result.dtype)
            func.prepared_async_call(
                self._grid,
                self._block,
                None,
                self.gpudata,
                result.gpudata,
                self.mem_size,
            )

            return result
        else:
            return self

    conjugate = conj

    # }}}

    # {{{ rich comparisons

    __eq__ = _make_binary_op("==")
    __ne__ = _make_binary_op("!=")
    __le__ = _make_binary_op("<=")
    __ge__ = _make_binary_op(">=")
    __lt__ = _make_binary_op("<")
    __gt__ = _make_binary_op(">")

    # }}}


# }}}


# {{{ creation helpers


def to_gpu(ary, allocator=drv.mem_alloc):
    """converts a numpy array to a GPUArray"""
    result = GPUArray(ary.shape, ary.dtype, allocator, strides=_compact_strides(ary))
    result.set(ary)
    return result


def to_gpu_async(ary, allocator=drv.mem_alloc, stream=None):
    """converts a numpy array to a GPUArray"""
    result = GPUArray(ary.shape, ary.dtype, allocator, strides=_compact_strides(ary))
    result.set_async(ary, stream)
    return result


empty = GPUArray


def zeros(shape, dtype=np.float64, allocator=drv.mem_alloc, order="C"):
    """Returns an array of the given shape and dtype filled with 0's."""
    result = GPUArray(shape, dtype, allocator, order=order)
    zero = np.zeros((), dtype)
    result.fill(zero)
    return result


def ones(shape, dtype=np.float64, allocator=drv.mem_alloc, order="C"):
    """Returns an array of the given shape and dtype filled with 1's."""
    result = GPUArray(shape, dtype, allocator, order=order)
    one = np.ones((), dtype)
    result.fill(one)
    return result


def _array_like_helper(other_ary, dtype, order):
    """Set order, strides, dtype as in numpy's zero_like. """
    strides = None
    if order == "A":
        if other_ary.flags.f_contiguous and not other_ary.flags.c_contiguous:
            order = "F"
        else:
            order = "C"
    elif order == "K":
        if other_ary.flags.c_contiguous or (other_ary.ndim <= 1):
            order = "C"
        elif other_ary.flags.f_contiguous:
            order = "F"
        else:
            # array_like routines only return positive strides
            strides = [np.abs(s) for s in other_ary.strides]
            if dtype is not None and dtype != other_ary.dtype:
                # scale strides by itemsize when dtype is not the same
                itemsize = other_ary.nbytes // other_ary.size
                itemsize_ratio = np.dtype(dtype).itemsize / itemsize
                strides = [int(s * itemsize_ratio) for s in strides]
    elif order not in ["C", "F"]:
        raise ValueError("Unsupported order: %r" % order)
    if dtype is None:
        dtype = other_ary.dtype
    return dtype, order, strides


def empty_like(other_ary, dtype=None, order="K"):
    dtype, order, strides = _array_like_helper(other_ary, dtype, order)
    result = GPUArray(
        other_ary.shape, dtype, other_ary.allocator, order=order, strides=strides
    )
    return result


def zeros_like(other_ary, dtype=None, order="K"):
    dtype, order, strides = _array_like_helper(other_ary, dtype, order)
    result = GPUArray(
        other_ary.shape, dtype, other_ary.allocator, order=order, strides=strides
    )
    zero = np.zeros((), result.dtype)
    result.fill(zero)
    return result


def ones_like(other_ary, dtype=None, order="K"):
    dtype, order, strides = _array_like_helper(other_ary, dtype, order)
    result = GPUArray(
        other_ary.shape, dtype, other_ary.allocator, order=order, strides=strides
    )
    one = np.ones((), result.dtype)
    result.fill(one)
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

    if isinstance(args[-1], np.dtype):
        inf.dtype = args[-1]
        args = args[:-1]
        explicit_dtype = True

    argc = len(args)
    if argc == 0:
        raise ValueError("stop argument required")
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
        raise ValueError("too many arguments")

    admissible_names = ["start", "stop", "step", "dtype"]
    for k, v in kwargs.items():
        if k in admissible_names:
            if getattr(inf, k) is None:
                setattr(inf, k, v)
                if k == "dtype":
                    explicit_dtype = True
            else:
                raise ValueError("may not specify '%s' by position and keyword" % k)
        else:
            raise ValueError("unexpected keyword argument '%s'" % k)

    if inf.start is None:
        inf.start = 0
    if inf.step is None:
        inf.step = 1
    if inf.dtype is None:
        inf.dtype = np.array([inf.start, inf.stop, inf.step]).dtype

    # actual functionality ----------------------------------------------------
    dtype = np.dtype(inf.dtype)
    start = dtype.type(inf.start)
    step = dtype.type(inf.step)
    stop = dtype.type(inf.stop)

    if not explicit_dtype and dtype != np.float32:
        from warnings import warn

        warn(
            "behavior change: arange guessed dtype other than float32. "
            "suggest specifying explicit dtype."
        )

    from math import ceil

    size = int(ceil((stop - start) / step))

    result = GPUArray((size,), dtype)

    func = elementwise.get_arange_kernel(dtype)
    func.prepared_async_call(
        result._grid,
        result._block,
        kwargs.get("stream"),
        result.gpudata,
        start,
        step,
        size,
    )

    return result


# }}}


def _compact_strides(a):
    # Compute strides to have same order as self, but packed
    info = sorted(
        (a.strides[axis], a.shape[axis], axis) for axis in range(len(a.shape))
    )

    strides = [None] * len(a.shape)
    stride = a.dtype.itemsize
    for _, dim, axis in info:
        strides[axis] = stride
        stride *= dim
    return strides


def _memcpy_discontig(dst, src, async_=False, stream=None):
    """Copy the contents of src into dst.

    The two arrays should have the same dtype, shape, and order, but
    not necessarily the same strides. There may be up to _two_
    axes along which either `src` or `dst` is not contiguous.
    """

    if not isinstance(src, (GPUArray, np.ndarray)):
        raise TypeError("src must be GPUArray or ndarray")
    if not isinstance(dst, (GPUArray, np.ndarray)):
        raise TypeError("dst must be GPUArray or ndarray")
    if src.shape != dst.shape:
        raise ValueError("src and dst must be same shape")
    if src.dtype != dst.dtype:
        raise TypeError("src and dst must have same dtype")

    # ndarray -> ndarray
    if isinstance(src, np.ndarray) and isinstance(dst, np.ndarray):
        dst[...] = src
        return

    if src.flags.forc and dst.flags.forc:
        shape = [src.size]
        src_strides = dst_strides = [src.dtype.itemsize]
    else:
        # put src in Fortran order (which should put dst in Fortran order too)
        # and remove singleton axes
        src_info = sorted(
            (src.strides[axis], axis)
            for axis in range(len(src.shape))
            if src.shape[axis] > 1
        )
        axes = [axis for _, axis in src_info]
        shape = [src.shape[axis] for axis in axes]
        src_strides = [src.strides[axis] for axis in axes]
        dst_strides = [dst.strides[axis] for axis in axes]

        # copy functions require contiguity in minor axis, so add new axis if needed
        if (
            len(shape) == 0
            or src_strides[0] != src.dtype.itemsize
            or dst_strides[0] != dst.dtype.itemsize
        ):
            shape[0:0] = [1]
            src_strides[0:0] = [0]
            dst_strides[0:0] = [0]
            axes[0:0] = [np.newaxis]

        # collapse contiguous dimensions
        # and check that dst is in same order as src
        i = 1
        while i < len(shape):
            if dst_strides[i] < dst_strides[i - 1]:
                raise ValueError("src and dst must have same order")
            if (
                src_strides[i - 1] * shape[i - 1] == src_strides[i]
                and dst_strides[i - 1] * shape[i - 1] == dst_strides[i]
            ):
                shape[i - 1:i + 1] = [shape[i - 1] * shape[i]]
                del src_strides[i]
                del dst_strides[i]
                del axes[i]
            else:
                i += 1

    if len(shape) <= 1:
        if isinstance(src, GPUArray):
            if isinstance(dst, GPUArray):
                if async_:
                    drv.memcpy_dtod_async(
                        dst.gpudata, src.gpudata, src.nbytes, stream=stream
                    )
                else:
                    drv.memcpy_dtod(dst.gpudata, src.gpudata, src.nbytes)
            else:
                # The arrays might be contiguous in the sense of
                # having no gaps, but the axes could be transposed
                # so that the order is neither Fortran or C.
                # So, we attempt to get a contiguous view of dst.
                dst = _as_strided(dst, shape=(dst.size,), strides=(dst.dtype.itemsize,))
                if async_:
                    drv.memcpy_dtoh_async(dst, src.gpudata, stream=stream)
                else:
                    drv.memcpy_dtoh(dst, src.gpudata)
        else:
            src = _as_strided(src, shape=(src.size,), strides=(src.dtype.itemsize,))
            if async_:
                drv.memcpy_htod_async(dst.gpudata, src, stream=stream)
            else:
                drv.memcpy_htod(dst.gpudata, src)
        return

    if len(shape) == 2:
        copy = drv.Memcpy2D()
    elif len(shape) == 3:
        copy = drv.Memcpy3D()
    else:
        raise ValueError(
            "more than 2 discontiguous axes not supported {}".format(tuple(sorted(axes)))
        )

    if isinstance(src, GPUArray):
        copy.set_src_device(src.gpudata)
    else:
        copy.set_src_host(src)

    if isinstance(dst, GPUArray):
        copy.set_dst_device(dst.gpudata)
    else:
        copy.set_dst_host(dst)

    copy.width_in_bytes = src.dtype.itemsize * shape[0]

    copy.src_pitch = src_strides[1]
    copy.dst_pitch = dst_strides[1]
    copy.height = shape[1]

    if len(shape) == 2:
        if async_:
            copy(stream)
        else:
            copy(aligned=True)

    else:  # len(shape) == 3
        if src_strides[2] % src_strides[1] != 0:
            raise RuntimeError("src's major stride must be a multiple of middle stride")
        copy.src_height = src_strides[2] // src_strides[1]

        if dst_strides[2] % dst_strides[1] != 0:
            raise RuntimeError("dst's major stride must be a multiple of middle stride")
        copy.dst_height = dst_strides[2] // dst_strides[1]

        copy.depth = shape[2]
        if async_:
            copy(stream)
        else:
            copy()


# {{{ pickle support

copyreg.pickle(GPUArray, lambda data: (to_gpu, (data.get(),)), to_gpu)

# }}}


# {{{ take/put


def take(a, indices, out=None, stream=None):
    if out is None:
        out = GPUArray(indices.shape, a.dtype, a.allocator)

    assert len(indices.shape) == 1

    func, tex_src = elementwise.get_take_kernel(a.dtype, indices.dtype)
    a.bind_to_texref_ext(tex_src[0], allow_double_hack=True, allow_complex_hack=True)

    func.prepared_async_call(
        out._grid, out._block, stream, indices.gpudata, out.gpudata, indices.size
    )

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
        out = [GPUArray(indices.shape, a_dtype, a_allocator) for i in range(vec_count)]
    else:
        if len(out) != len(arrays):
            raise ValueError("out and arrays must have the same length")

    chunk_size = _builtin_min(vec_count, 20)

    def make_func_for_chunk_size(chunk_size):
        return elementwise.get_take_kernel(a_dtype, indices.dtype, vec_count=chunk_size)

    func, tex_src = make_func_for_chunk_size(chunk_size)

    for start_i in range(0, len(arrays), chunk_size):
        chunk_slice = slice(start_i, start_i + chunk_size)

        if start_i + chunk_size > vec_count:
            func, tex_src = make_func_for_chunk_size(vec_count - start_i)

        for i, a in enumerate(arrays[chunk_slice]):
            a.bind_to_texref_ext(tex_src[i], allow_double_hack=True)

        func.prepared_async_call(
            indices._grid,
            indices._block,
            stream,
            indices.gpudata,
            *([o.gpudata for o in out[chunk_slice]] + [indices.size])
        )

    return out


def multi_take_put(
    arrays,
    dest_indices,
    src_indices,
    dest_shape=None,
    out=None,
    stream=None,
    src_offsets=None,
):
    if not len(arrays):
        return []

    from pytools import single_valued

    a_dtype = single_valued(a.dtype for a in arrays)
    a_allocator = arrays[0].allocator

    vec_count = len(arrays)

    if out is None:
        out = [GPUArray(dest_shape, a_dtype, a_allocator) for i in range(vec_count)]
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
        return elementwise.get_take_put_kernel(
            a_dtype,
            src_indices.dtype,
            with_offsets=src_offsets is not None,
            vec_count=chunk_size,
        )

    func, tex_src = make_func_for_chunk_size(chunk_size)

    for start_i in range(0, len(arrays), chunk_size):
        chunk_slice = slice(start_i, start_i + chunk_size)

        if start_i + chunk_size > vec_count:
            func, tex_src = make_func_for_chunk_size(vec_count - start_i)

        for src_tr, a in zip(tex_src, arrays[chunk_slice]):
            a.bind_to_texref_ext(src_tr, allow_double_hack=True)

        func.prepared_async_call(
            src_indices._grid,
            src_indices._block,
            stream,
            dest_indices.gpudata,
            src_indices.gpudata,
            *(
                [o.gpudata for o in out[chunk_slice]]
                + src_offsets_list[chunk_slice]
                + [src_indices.size]
            )
        )

    return out


def multi_put(arrays, dest_indices, dest_shape=None, out=None, stream=None):
    if not len(arrays):
        return []

    from pytools import single_valued

    a_dtype = single_valued(a.dtype for a in arrays)
    a_allocator = arrays[0].allocator

    vec_count = len(arrays)

    if out is None:
        out = [GPUArray(dest_shape, a_dtype, a_allocator) for i in range(vec_count)]
    else:
        if a_dtype != single_valued(o.dtype for o in out):
            raise TypeError("arrays and out must have the same dtype")
        if len(out) != vec_count:
            raise ValueError("out and arrays must have the same length")

    if len(dest_indices.shape) != 1:
        raise ValueError("src_indices must be 1D")

    chunk_size = _builtin_min(vec_count, 10)

    def make_func_for_chunk_size(chunk_size):
        return elementwise.get_put_kernel(
            a_dtype, dest_indices.dtype, vec_count=chunk_size
        )

    func = make_func_for_chunk_size(chunk_size)

    for start_i in range(0, len(arrays), chunk_size):
        chunk_slice = slice(start_i, start_i + chunk_size)

        if start_i + chunk_size > vec_count:
            func = make_func_for_chunk_size(vec_count - start_i)

        func.prepared_async_call(
            dest_indices._grid,
            dest_indices._block,
            stream,
            dest_indices.gpudata,
            *(
                [o.gpudata for o in out[chunk_slice]]
                + [i.gpudata for i in arrays[chunk_slice]]
                + [dest_indices.size]
            )
        )

    return out


# }}}


# {{{ shape manipulation

def concatenate(arrays, axis=0, allocator=None):
    """
    Join a sequence of arrays along an existing axis.
    :arg arrays: A sequnce of :class:`GPUArray`.
    :arg axis: Index of the dimension of the new axis in the result array.
        Can be -1, for the new axis to be last dimension.
    :returns: :class:`GPUArray`
    """
    # implementation is borrowed from pyopencl.array.concatenate()
    # {{{ find properties of result array

    shape = None

    def shape_except_axis(ary: GPUArray):
        return ary.shape[:axis] + ary.shape[axis+1:]

    for i_ary, ary in enumerate(arrays):
        allocator = allocator or ary.allocator

        if shape is None:
            # first array
            shape = list(ary.shape)

        else:
            if len(ary.shape) != len(shape):
                raise ValueError("%d'th array has different number of axes "
                        "(should have %d, has %d)"
                        % (i_ary, len(ary.shape), len(shape)))

            if (ary.ndim != arrays[0].ndim
                    or shape_except_axis(ary) != shape_except_axis(arrays[0])):
                raise ValueError("%d'th array has residual not matching "
                        "other arrays" % i_ary)

            shape[axis] += ary.shape[axis]

    # }}}

    shape = tuple(shape)
    dtype = np.result_type(*(ary.dtype for ary in arrays))
    result = empty(shape, dtype, allocator=allocator)

    full_slice = (slice(None),) * len(shape)

    base_idx = 0
    for ary in arrays:
        my_len = ary.shape[axis]
        result[full_slice[:axis] + (slice(base_idx, base_idx+my_len),) + full_slice[axis+1:]] = ary
        base_idx += my_len

    return result


def stack(arrays, axis=0, allocator=None):
    """
    Join a sequence of arrays along a new axis.
    :arg arrays: A sequnce of :class:`GPUArray`.
    :arg axis: Index of the dimension of the new axis in the result array.
        Can be -1, for the new axis to be last dimension.
    :returns: :class:`GPUArray`
    """
    # implementation is borrowed from pyopencl.array.stack()
    allocator = allocator or arrays[0].allocator

    if not arrays:
        raise ValueError("need at least one array to stack")

    input_shape = arrays[0].shape
    input_ndim = arrays[0].ndim
    axis = input_ndim if axis == -1 else axis

    import builtins
    if not builtins.all(ary.shape == input_shape for ary in arrays[1:]):
        raise ValueError("arrays must have the same shape")

    if not (0 <= axis <= input_ndim):
        raise ValueError("invalid axis")

    result_shape = input_shape[:axis] + (len(arrays),) + input_shape[axis:]
    result = empty(shape=result_shape,
            dtype=np.result_type(*(ary.dtype for ary in arrays)),
            allocator=allocator, order="C" if axis == 0 else "F")

    for i, ary in enumerate(arrays):

        idx = (slice(None),)*axis + (i,) + (slice(None),)*(input_ndim-axis)
        result[idx] = ary

    return result


def transpose(a, axes=None):
    """Permute the dimensions of an array.

    :arg a: :class:`GPUArray`
    :arg axes: list of ints, optional.
        By default, reverse the dimensions, otherwise permute the axes
        according to the values given.

    :returns: :class:`GPUArray` A view of the array with its axes permuted.

    .. versionadded:: 2015.2
    """
    return a.transpose(axes)


def reshape(a, *shape, **kwargs):
    """Gives a new shape to an array without changing its data.

    .. versionadded:: 2015.2
    """

    return a.reshape(*shape, **kwargs)


# }}}


# {{{ conditionals


def if_positive(criterion, then_, else_, out=None, stream=None):
    if not (criterion.shape == then_.shape == else_.shape):
        raise ValueError("shapes do not match")

    if not (then_.dtype == else_.dtype):
        raise ValueError("dtypes do not match")

    func = elementwise.get_if_positive_kernel(criterion.dtype, then_.dtype)

    if out is None:
        out = empty_like(then_)

    func.prepared_async_call(
        criterion._grid,
        criterion._block,
        stream,
        criterion.gpudata,
        then_.gpudata,
        else_.gpudata,
        out.gpudata,
        criterion.size,
    )

    return out


def where(criterion, then_, else_, out=None, stream=None):
    if (criterion.shape != then_.shape != else_.shape):
        raise NotImplementedError("shape broadcast not implemented")

    if (then_.dtype != else_.dtype):
        raise NotImplementedError("dtype broadcast not implemented")

    func = elementwise.get_where_kernel(criterion.dtype, then_.dtype)

    if out is None:
        out = empty_like(then_)

    func.prepared_async_call(
        criterion._grid,
        criterion._block,
        stream,
        criterion.gpudata,
        then_.gpudata,
        else_.gpudata,
        out.gpudata,
        criterion.size,
    )

    return out


def _make_binary_minmax_func(which):
    def f(a, b, out=None, stream=None):
        if isinstance(a, GPUArray) and isinstance(b, GPUArray):
            if out is None:
                out = empty_like(a)
            func = elementwise.get_binary_minmax_kernel(
                which, a.dtype, b.dtype, out.dtype, use_scalar=False
            )

            func.prepared_async_call(
                a._grid, a._block, stream, a.gpudata, b.gpudata, out.gpudata, a.size
            )
        elif isinstance(a, GPUArray):
            if out is None:
                out = empty_like(a)
            func = elementwise.get_binary_minmax_kernel(
                which, a.dtype, a.dtype, out.dtype, use_scalar=True
            )
            func.prepared_async_call(
                a._grid, a._block, stream, a.gpudata, b, out.gpudata, a.size
            )
        else:  # assuming b is a GPUArray
            if out is None:
                out = empty_like(b)
            func = elementwise.get_binary_minmax_kernel(
                which, b.dtype, b.dtype, out.dtype, use_scalar=True
            )
            # NOTE: we switch the order of a and b here!
            func.prepared_async_call(
                b._grid, b._block, stream, b.gpudata, a, out.gpudata, b.size
            )
        return out

    return f


minimum = _make_binary_minmax_func("min")
maximum = _make_binary_minmax_func("max")

# }}}


# {{{ reductions


def sum(a, dtype=None, stream=None, allocator=None):
    from pycuda.reduction import get_sum_kernel

    krnl = get_sum_kernel(dtype, a.dtype)
    return krnl(a, stream=stream, allocator=allocator)


def any(a, stream=None, allocator=None):
    from pycuda.reduction import get_any_kernel

    krnl = get_any_kernel(np.dtype(bool), a.dtype)
    return krnl(a, stream=stream, allocator=allocator)


def all(a, stream=None, allocator=None):
    from pycuda.reduction import get_all_kernel

    krnl = get_all_kernel(np.dtype(bool), a.dtype)
    return krnl(a, stream=stream, allocator=allocator)


def subset_sum(subset, a, dtype=None, stream=None, allocator=None):
    from pycuda.reduction import get_subset_sum_kernel

    krnl = get_subset_sum_kernel(dtype, subset.dtype, a.dtype)
    return krnl(subset, a, stream=stream, allocator=allocator)


def dot(a, b, dtype=None, stream=None, allocator=None):
    from pycuda.reduction import get_dot_kernel

    if dtype is None:
        dtype = _get_common_dtype(a, b)
    krnl = get_dot_kernel(dtype, a.dtype, b.dtype)
    return krnl(a, b, stream=stream, allocator=allocator)


def subset_dot(subset, a, b, dtype=None, stream=None, allocator=None):
    from pycuda.reduction import get_subset_dot_kernel

    krnl = get_subset_dot_kernel(dtype, subset.dtype, a.dtype, b.dtype)
    return krnl(subset, a, b, stream=stream, allocator=allocator)


def _make_minmax_kernel(what):
    def f(a, stream=None):
        from pycuda.reduction import get_minmax_kernel

        krnl = get_minmax_kernel(what, a.dtype)
        return krnl(a, stream=stream)

    return f


_builtin_min = min
_builtin_max = max
min = _make_minmax_kernel("min")
max = _make_minmax_kernel("max")


def _make_subset_minmax_kernel(what):
    def f(subset, a, stream=None):
        from pycuda.reduction import get_subset_minmax_kernel

        krnl = get_subset_minmax_kernel(what, a.dtype, subset.dtype)
        return krnl(subset, a, stream=stream)

    return f


subset_min = _make_subset_minmax_kernel("min")
subset_max = _make_subset_minmax_kernel("max")

# }}}


# {{{ logical ops

def _logical_op(x1, x2, out, allocator, operator):
    assert operator in ["&&", "||"]
    allocator = (
        allocator
        or getattr(x1, "allocator", None)
        or getattr(x2, "allocator", None)
        or drv.mem_alloc)

    if np.isscalar(x1) and np.isscalar(x2):
        if out is None:
            out = empty(shape=(), dtype=np.bool_, allocator=allocator)

        if operator == "&&":
            out[:] = np.logical_and(x1, x2)
        else:
            out[:] = np.logical_or(x1, x2)
    elif np.isscalar(x1) or np.isscalar(x2):
        scalar_arg, = [x for x in (x1, x2) if np.isscalar(x)]
        ary_arg, = [x for x in (x1, x2) if not np.isscalar(x)]
        if not isinstance(ary_arg, GPUArray):
            raise ValueError("logical_and can take either scalar or GPUArrays"
                             " as inputs")

        out = out or ary_arg._new_like_me(dtype=np.bool_)

        assert out.shape == ary_arg.shape and out.dtype == np.bool_

        func = elementwise.get_scalar_op_kernel(ary_arg.dtype,
                                                np.dtype(type(scalar_arg)),
                                                out.dtype,
                                                operator)

        func.prepared_async_call(out._grid, out._block,
                                 None,
                                 ary_arg.gpudata,
                                 scalar_arg,
                                 out.gpudata,
                                 out.mem_size)
    else:
        if not (isinstance(x1, GPUArray) and isinstance(x2, GPUArray)):
            raise ValueError("logical_and can take either scalar or GPUArrays"
                             " as inputs")
        if x1.shape != x2.shape:
            raise NotImplementedError("Broadcasting not supported")

        if out is None:
            out = x1._new_like_me(dtype=np.bool_)

        assert out.shape == x1.shape and out.dtype == np.bool_

        func = elementwise.get_binary_op_kernel(
            x1.dtype, x2.dtype, out.dtype, operator
        )
        func.prepared_async_call(out._grid, out._block,
                                 None,
                                 x1.gpudata,
                                 x2.gpudata,
                                 out.gpudata,
                                 out.mem_size)

    return out


def logical_and(x1, x2, /, out=None, *, allocator=None):
    return _logical_op(x1, x2, out, allocator, "&&")


def logical_or(x1, x2, /, out=None, *, allocator=None):
    return _logical_op(x1, x2, out, allocator, "||")


def logical_not(x, /, out=None, *, allocator=drv.mem_alloc):
    if np.isscalar(x):
        out = out or empty(shape=(), dtype=np.bool_, allocator=allocator)
        out[:] = np.logical_not(x)
    else:
        out = out or empty(shape=x.shape, dtype=np.bool_, allocator=allocator)
        func = elementwise.get_logical_not_kernel(x.dtype, out.dtype)
        func.prepared_async_call(out._grid, out._block,
                                 None,
                                 x.gpudata,
                                 out.gpudata,
                                 out.mem_size)

    return out

# }}}

# vim: foldmethod=marker
