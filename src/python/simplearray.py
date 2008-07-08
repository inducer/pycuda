import pycuda.driver as drv
import pycuda.gpuarray as gpuarray

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
    
def to_gpu(ary, stream=None):
    """converts a numpy array to a GPUArray"""
    result = SimpleArray(ary.shape, ary.dtype)
    result.set(ary, stream)
    return result




empty = SimpleArray

def zeros(shape, dtype, stream=None):
    """creates an array of the given size and fills it with 0's"""
    result = SimpleArray(shape, dtype, stream)
    result.fill(0)
    return result