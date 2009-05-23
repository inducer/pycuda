import pycuda.gpuarray as gpuarray
import pycuda.elementwise as elementwise
import numpy

def _make_unary_array_func(name):
    def f(array, stream=None):
        result = array._new_like_me()
        
        if array.dtype == numpy.float32:
            func_name = name + "f"
        else:
            func_name = name

        func = elementwise.get_unary_func_kernel(func_name, array.dtype)
        func.set_block_shape(*array._block)
        func.prepared_async_call(array._grid, stream,
                array.gpudata, result.gpudata, array.mem_size)
        
        return result
    return f

fabs = _make_unary_array_func("fabs")
ceil = _make_unary_array_func("ceil")
floor = _make_unary_array_func("floor")
exp = _make_unary_array_func("exp")
log = _make_unary_array_func("log")
log10 = _make_unary_array_func("log10")
sqrt = _make_unary_array_func("sqrt")

sin = _make_unary_array_func("sin")
cos = _make_unary_array_func("cos")
tan = _make_unary_array_func("tan")
asin = _make_unary_array_func("asin")
acos = _make_unary_array_func("acos")
atan = _make_unary_array_func("atan")

sinh = _make_unary_array_func("sinh")
cosh = _make_unary_array_func("cosh")
tanh = _make_unary_array_func("tanh")
        
def fmod(arg, mod, stream=None):
    """Return the floating point remainder of the division `arg/mod`,
    for each element in `arg` and `mod`."""
    result = gpuarray.GPUArray(arg.shape, arg.dtype)
    
    func = elementwise.get_fmod_kernel()
    func.set_block_shape(*arg._block)
    func.prepared_async_call(arg._grid, stream,
            arg.gpudata, mod.gpudata, result.gpudata, arg.mem_size)
    
    return result

def frexp(arg, stream=None):
    """Return a tuple `(significands, exponents)` such that 
    `arg == significand * 2**exponent`.
    """
    sig = gpuarray.GPUArray(arg.shape, arg.dtype)
    expt = gpuarray.GPUArray(arg.shape, arg.dtype)
    
    func = elementwise.get_frexp_kernel()
    func.set_block_shape(*arg._block)
    func.prepared_async_call(arg._grid, stream,
            arg.gpudata, sig.gpudata, expt.gpudata, arg.mem_size)
    
    return sig, expt
    
def ldexp(significand, exponent, stream=None):
    """Return a new array of floating point values composed from the
    entries of `significand` and `exponent`, paired together as
    `result = significand * 2**exponent`.
    """
    result = gpuarray.GPUArray(significand.shape, significand.dtype)
    
    func = elementwise.get_ldexp_kernel()
    func.set_block_shape(*significand._block)
    func.prepared_async_call(significand._grid, stream,
            significand.gpudata, exponent.gpudata, result.gpudata, 
            significand.mem_size)
    
    return result
        
def modf(arg, stream=None):
    """Return a tuple `(fracpart, intpart)` of arrays containing the
    integer and fractional parts of `arg`. 
    """

    intpart = gpuarray.GPUArray(arg.shape, arg.dtype)
    fracpart = gpuarray.GPUArray(arg.shape, arg.dtype)
    
    func = elementwise.get_modf_kernel()
    func.set_block_shape(*arg._block),
    func.prepared_async_call(arg._grid, stream,
            arg.gpudata, intpart.gpudata, fracpart.gpudata,
            arg.mem_size)
    
    return fracpart, intpart
