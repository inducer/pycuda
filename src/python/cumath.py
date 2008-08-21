import pycuda.gpuarray as gpuarray
import pycuda._kernel as _kernel
import numpy
import math

def add(x,y):
    """adds something"""
    return x + y

def substract(x,y):
    """substract something"""
    return x - y

def multiply(x,y):
    """multiply something"""
    return x * y

def divide(x,y):
    """divides something"""
    return x / y


def ceil(array):
    """executes the ceil function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        _kernel._get_ceil_kernel()(array.gpudata,
                result.gpudata, numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        return math.ceil(array)
    
def fabs(array):
    """executes the fabs function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        return abs(result)
    else:
        return math.fabs(array)

def floor(array):
    """executes the floor function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        _kernel._get_floor_kernel()(array.gpudata,
                result.gpudata, numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        return math.floor(array)

def fmod(array,mod):
    """executes the fmod function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        _kernel._get_fmod_kernel()(array.gpudata,
                result.gpudata,numpy.float32(mod), numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        return math.fmod(array, mod)

def frexp(array):
    """executes the frexp function on the gpu for all elements in the given array::
    
       it return a tuple with two gpu arrays which contains the calculates results
    """
    if isinstance(array, gpuarray.GPUArray):
        first = gpuarray.GPUArray(array.shape, array.dtype)
        second = gpuarray.GPUArray(array.shape, array.dtype)
        
        _kernel._get_frexp_kernel()(array.gpudata,
                first.gpudata,second.gpudata,numpy.int32(array.size),
                **first._kernel_kwargs)
        
        return (first,second)
    else:
        return math.frexp(array)
    
def ldexp(array, i):
    """executes the ldexp function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        _kernel._get_ldexp_kernel()(array.gpudata,
                result.gpudata,numpy.float32(i), numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        return math.ldexp(array, i)
        
def modf(array):
    """executes the modf function on the gpu for all elements in the given array::
    
       it return a tuple with two gpu arrays which contains the calculates results
    """
    if isinstance(array, gpuarray.GPUArray):
        first = gpuarray.GPUArray(array.shape, array.dtype)
        second = gpuarray.GPUArray(array.shape, array.dtype)
        
        _kernel._get_modf_kernel()(array.gpudata,
                first.gpudata,second.gpudata,numpy.int32(array.size),
                **first._kernel_kwargs)
        
        return (first,second)
    else:
        return math.modf(array)

def exp(array):
    """executes the exp function on the gpu for all elements in the given array::
    
       it seems that the original python math function is much more precise than the float
       based version in C.
    """
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        _kernel._get_exp_kernel()(array.gpudata,
                result.gpudata,numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        return math.exp(array)

def log(array):
    """executes the log function on the gpu for all elements in the given array::
    
       we only support the base of two since the internal cuda function does not
       allow to specify the base
    """
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        _kernel._get_log_kernel()(array.gpudata,
                result.gpudata,numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        return math.log(array)
    
def log10(array):
    """executes the log10 function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        _kernel._get_log10_kernel()(array.gpudata,
                result.gpudata,numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        return math.log10(array)

def pow(x,y):
    """executes the pow function on the gpu for all elements in the given array"""
    if isinstance(x, gpuarray.GPUArray):
        return x.__pow__(y)
    else:
        return math.pow(x, y)

def sqrt(array):
    """executes the sqrt function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        _kernel._get_sqrt_kernel()(array.gpudata,
                result.gpudata,numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        return math.sqrt(array)
    
def acos(array):
    """executes the acos function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        _kernel._get_acos_kernel()(array.gpudata,
                result.gpudata,numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        return math.acos(array)


def cos(array):
    """executes the cos function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        _kernel._get_cos_kernel()(array.gpudata,
                result.gpudata,numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        return math.cos(array)


def tan(array):
    """executes the tan function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        _kernel._get_tan_kernel()(array.gpudata,
                result.gpudata,numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        return math.tan(array)

def atan(array):
    """executes the cos function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        _kernel._get_atan_kernel()(array.gpudata,
                result.gpudata,numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        return math.atan(array)


def asin(array):
    """executes the asin function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        _kernel._get_asin_kernel()(array.gpudata,
                result.gpudata,numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        return math.asin(array)


def sin(array):
    """executes the sin function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        _kernel._get_sin_kernel()(array.gpudata,
                result.gpudata,numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        return math.sin(array)
        
def sinh(array):
    """executes the sinh function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        _kernel._get_sinh_kernel()(array.gpudata,
                result.gpudata,numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        return math.sinh(array)

def tanh(array):
    """executes the tah function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        _kernel._get_tanh_kernel()(array.gpudata,
                result.gpudata,numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        return math.tanh(array)

def cosh(array):
    """executes the cosh function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        _kernel._get_cosh_kernel()(array.gpudata,
                result.gpudata,numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        return math.cosh(array)

def degrees(array):
    """executes the cosh function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        _kernel._get_degress_kernel()(array.gpudata,
                result.gpudata,numpy.float32(pi),numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        return math.degrees(array)


def radians(array):
    """executes the cosh function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        _kernel._get_radians_kernel()(array.gpudata,
                result.gpudata,numpy.float32(pi),numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        return math.radians(array)
    
#cconstant pi
pi = 3.14159265358979323846

#constant e
e = 2.7182818284590451
