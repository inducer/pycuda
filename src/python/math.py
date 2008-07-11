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

