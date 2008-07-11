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
        
        kernel._get_ceil_kernel()(array.gpudata,
                result.gpudata, numpy.int32(array.size),
                **result._kernel_kwargs)
        
        return result
    else:
        return math.fabs(array)