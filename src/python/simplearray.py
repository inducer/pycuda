import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import numpy

#if the context was not created, created it now
if not drv.was_context_created():
    drv.init()
    assert drv.Device.count() >= 1
    dev = drv.Device(0)
    ctx = dev.make_context()

class SimpleArray(gpuarray.GPUArray):
    """A simplified class to work with GPU arrays.
    
    Basically it initializes the context for you and makes the live easier.
    But it will always work on the first device found!
    
    """

    def __init__(self, shape, dtype, stream=None):
        """call to the top level constructor"""
        gpuarray.GPUArray.__init__(self,shape,dtype,stream)
    
    
def to_gpu_from_numpy(ary, stream=None):
    """converts a numpy array to a GPUArray"""
    result = SimpleArray(ary.shape,ary.dtype,stream)
    result.set(ary, stream)
    return result

    
def to_gpu(ary, stream=None):
    """converts a numpy array to a GPUArray"""
    return to_gpu_from_numpy(ary,stream)

    
def to_gpu_from_list(list, stream=None):
    """converts a list to a GPUArray"""
    return to_gpu_from_numpy(numpy.array(list).astype(numpy.float32),stream)


empty = SimpleArray

def list(size,value=0):
    """creates a list of the given size"""
    return fill((size,1),value)


def matrix(width,height,value=0):
    """creates a matrix of the given size"""
    return fill((width,height),value)


def fill(shape,value, dtype=numpy.float32, stream=None):
    """creates an array of the given shape and fills it with the data"""
    result = SimpleArray(shape, dtype, stream)
    result.fill(value)
    return result


def zeros(shape, dtype=numpy.float32, stream=None):
    """creates an array of the given size and fills it with 0's"""
    return fill(shape,0,dtype,stream)
