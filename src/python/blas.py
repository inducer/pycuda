import numpy
import pyublas
import pycuda.rt
from pycuda._blas import *
init()

import atexit
atexit.register(shutdown)




class Vector:
    def __init__(self, devptr, shape):
        self.devptr = devptr
        self.shape = shape

        if isinstance(self.devptr, DevicePtrFloat32):
            self.dtype = numpy.float32
        else:
            raise TypeError, "unknown device pointer type--huh?"

        from operator import mul
        self.size = reduce(mul, shape)

    @property
    def dtype(self):
        if isinstance(self.devptr, DevicePtrFloat32):
            return numpy.float32
        else:
            raise TypeError, "unknown device pointer type--huh?"

    def from_gpu(self):
        result = numpy.empty(self.shape, self.dtype)
        self.devptr.get(result)
        return result

    def __str__(self):
        return str(self.from_gpu())

    def __neg__(self):
        result = _empty_devptr(self.shape, self.dtype)
        copy(self.size, self.devptr, 1, result, 1)
        scal(self.size, -1, result, 1)
        return Vector(result, self.shape)

    def __add__(self, other):
        result = _empty_devptr(self.shape, self.dtype)
        copy(self.size, self.devptr, 1, result, 1)
        axpy(self.size, 1, other.devptr, 1, result, 1)
        return Vector(result, self.shape)

    def __sub__(self, other):
        result = _empty_devptr(self.shape, self.dtype)
        copy(self.size, self.devptr, 1, result, 1)
        axpy(self.size, -1, other.devptr, 1, result, 1)
        return Vector(result, self.shape)

    def __mul__(self, other):
        result = _empty_devptr(self.shape, self.dtype)
        copy(self.size, self.devptr, 1, result, 1)
        scal(self.size, other, result, 1)
        return Vector(result, self.shape)

    __rmul__ = __mul__

  


def _dtype_to_dptr_type(dtype):
    if dtype == numpy.float32:
        return DevicePtrFloat32
    else:
        raise TypeError, "invalid dtype"

def _empty_devptr(shape, dtype):
    from operator import mul
    size = reduce(mul, shape)
    return _dtype_to_dptr_type(dtype)(size)

def to_gpu(ary):
    devptr = _empty_devptr(ary.shape, ary.dtype)
    devptr.set(ary)
    return Vector(devptr, ary.shape)

def zeros(shape, dtype):
    return to_gpu(numpy.zeros(shape, dtype)) # FIXME: yuck! inefficient!

def ones(shape, dtype):
    return to_gpu(numpy.ones(shape, dtype)) # FIXME: yuck! inefficient!
