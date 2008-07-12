import pycuda.gpuarray as gpuarray
import pycuda.kernel as kernel
import numpy

def ceil(array):
    """executes the ceil function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        kernel._get_ceil_kernel()(array.gpudata,
                result.gpudata, numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        raise NotImplementedError, 'sorry only GPUArrays and subclasses are supported by this method'
    
def fabs(array):
    """executes the fabs function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        return abs(result)
    else:
        raise NotImplementedError, 'sorry only GPUArrays and subclasses are supported by this method'

def floor(array):
    """executes the floor function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        kernel._get_floor_kernel()(array.gpudata,
                result.gpudata, numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        raise NotImplementedError, 'sorry only GPUArrays and subclasses are supported by this method'
    

def fmod(array,mod):
    """executes the fmod function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        kernel._get_fmod_kernel()(array.gpudata,
                result.gpudata,numpy.float32(mod), numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        raise NotImplementedError, 'sorry only GPUArrays and subclasses are supported by this method'

def frexp(array):
    """executes the frexp function on the gpu for all elements in the given array::
    
       it return a tuple with two gpu arrays which contains the calculates results
    """
    if isinstance(array, gpuarray.GPUArray):
        first = gpuarray.GPUArray(array.shape, array.dtype)
        second = gpuarray.GPUArray(array.shape, array.dtype)
        
        kernel._get_frexp_kernel()(array.gpudata,
                first.gpudata,second.gpudata,numpy.int32(array.size),
                **first._kernel_kwargs)
        
        return (first,second)
    else:
        raise NotImplementedError, 'sorry only GPUArrays and subclasses are supported by this method'
    
def ldexp(array, i):
    """executes the ldexp function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        kernel._get_ldexp_kernel()(array.gpudata,
                result.gpudata,numpy.float32(i), numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        raise NotImplementedError, 'sorry only GPUArrays and subclasses are supported by this method'

def modf(array):
    """executes the modf function on the gpu for all elements in the given array::
    
       it return a tuple with two gpu arrays which contains the calculates results
    """
    if isinstance(array, gpuarray.GPUArray):
        first = gpuarray.GPUArray(array.shape, array.dtype)
        second = gpuarray.GPUArray(array.shape, array.dtype)
        
        kernel._get_modf_kernel()(array.gpudata,
                first.gpudata,second.gpudata,numpy.int32(array.size),
                **first._kernel_kwargs)
        
        return (first,second)
    else:
        raise NotImplementedError, 'sorry only GPUArrays and subclasses are supported by this method'

def exp(array):
    """executes the exp function on the gpu for all elements in the given array::
    
       it seems that the original python math function is much more precise than the float
       based version in C.
    """
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        kernel._get_exp_kernel()(array.gpudata,
                result.gpudata,numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        raise NotImplementedError, 'sorry only GPUArrays and subclasses are supported by this method'

def log(array):
    """executes the log function on the gpu for all elements in the given array::
    
       we only support the base of two since the internal cuda function does not
       allow to specify the base
    """
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        kernel._get_log_kernel()(array.gpudata,
                result.gpudata,numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        raise NotImplementedError, 'sorry only GPUArrays and subclasses are supported by this method'
    
def log10(array):
    """executes the log10 function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        kernel._get_log10_kernel()(array.gpudata,
                result.gpudata,numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        raise NotImplementedError, 'sorry only GPUArrays and subclasses are supported by this method'

def pow(x,y):
    """executes the pow function on the gpu for all elements in the given array"""
    if isinstance(x, gpuarray.GPUArray):
        return x.__pow__(y)
    else:
        raise NotImplementedError, 'sorry only GPUArrays and subclasses are supported by this method'

def sqrt(array):
    """executes the sqrt function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        kernel._get_sqrt_kernel()(array.gpudata,
                result.gpudata,numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        raise NotImplementedError, 'sorry only GPUArrays and subclasses are supported by this method'

def acos(array):
    """executes the acos function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        kernel._get_acos_kernel()(array.gpudata,
                result.gpudata,numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        raise NotImplementedError, 'sorry only GPUArrays and subclasses are supported by this method'


def cos(array):
    """executes the cos function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        kernel._get_cos_kernel()(array.gpudata,
                result.gpudata,numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        raise NotImplementedError, 'sorry only GPUArrays and subclasses are supported by this method'


def tan(array):
    """executes the tan function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        kernel._get_tan_kernel()(array.gpudata,
                result.gpudata,numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        raise NotImplementedError, 'sorry only GPUArrays and subclasses are supported by this method'


def atan(array):
    """executes the cos function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        kernel._get_atan_kernel()(array.gpudata,
                result.gpudata,numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        raise NotImplementedError, 'sorry only GPUArrays and subclasses are supported by this method'


def asin(array):
    """executes the asin function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        kernel._get_asin_kernel()(array.gpudata,
                result.gpudata,numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        raise NotImplementedError, 'sorry only GPUArrays and subclasses are supported by this method'


def sin(array):
    """executes the sin function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        kernel._get_sin_kernel()(array.gpudata,
                result.gpudata,numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        raise NotImplementedError, 'sorry only GPUArrays and subclasses are supported by this method'

def sinh(array):
    """executes the sinh function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        kernel._get_sinh_kernel()(array.gpudata,
                result.gpudata,numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        raise NotImplementedError, 'sorry only GPUArrays and subclasses are supported by this method'

def tanh(array):
    """executes the tah function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        kernel._get_tanh_kernel()(array.gpudata,
                result.gpudata,numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        raise NotImplementedError, 'sorry only GPUArrays and subclasses are supported by this method'

def cosh(array):
    """executes the cosh function on the gpu for all elements in the given array"""
    if isinstance(array, gpuarray.GPUArray):
        result = gpuarray.GPUArray(array.shape, array.dtype)
        
        kernel._get_cosh_kernel()(array.gpudata,
                result.gpudata,numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        raise NotImplementedError, 'sorry only GPUArrays and subclasses are supported by this method'

