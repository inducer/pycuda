from _driver import *
from pytools.diskdict import DiskDict
from pytools import memoize




class CompileError(Error):
    pass




class ArgumentHandler(object):
    def __init__(self, ary):
        self.array = ary
        self.dev_alloc = None

    def get_device_alloc(self):
        if self.dev_alloc is None:
            self.dev_alloc = mem_alloc_like(self.array)
        return self.dev_alloc

    def pre_call(self, stream):
        pass

class In(ArgumentHandler):
    def pre_call(self, stream):
        memcpy_htod(self.get_device_alloc(), self.array, stream)

class Out(ArgumentHandler):
    def post_call(self, stream):
        memcpy_dtoh(self.array, self.get_device_alloc(), stream)

class InOut(In, Out):
    pass





def _add_functionality():
    def device_get_attributes(dev):
        return dict((getattr(device_attribute, att), 
            dev.get_attribute(getattr(device_attribute, att))
            )
            for att in dir(device_attribute)
            if att[0].isupper())

    def function_param_set(func, *args):
        try:
            import numpy
        except ImportError:
            numpy = None

        handlers = []

        arg_data = []
        format = ""
        for i, arg in enumerate(args):
            if numpy is not None and isinstance(arg, numpy.number):
                arg_data.append(arg)
                format += arg.dtype.char
            elif isinstance(arg, (DeviceAllocation, PooledDeviceAllocation)):
                arg_data.append(int(arg))
                format += "P"
            elif isinstance(arg, ArgumentHandler):
                handlers.append(arg)
                arg_data.append(int(arg.get_device_alloc()))
                format += "P"
            elif isinstance(arg, buffer):
                arg_data.append(arg)
                format += "s"
            else:
                try:
                    gpudata = arg.gpudata
                except AttributeError:
                    raise TypeError("invalid type on parameter #%d (0-based)" % i)
                else:
                    # for gpuarrays
                    arg_data.append(int(gpudata))
                    format += "P"

        import struct
        buf = struct.pack(format, *arg_data)
        func.param_setv(0, buf)
        func.param_set_size(len(buf))

        return handlers

    def function_call(func, *args, **kwargs):
        grid = kwargs.pop("grid", (1,1))
        stream = kwargs.pop("stream", None)
        block = kwargs.pop("block", None)
        shared = kwargs.pop("shared", None)
        texrefs = kwargs.pop("texrefs", [])
        time_kernel = kwargs.pop("time_kernel", False)

        if kwargs:
            raise ValueError(
                    "extra keyword arguments: %s" 
                    % (",".join(kwargs.iterkeys())))

        if block is None:
            raise ValueError, "must specify block size"

        func.set_block_shape(*block)
        handlers = func.param_set(*args)
        if shared is not None:
            func.set_shared_size(shared)

        for handler in handlers:
            handler.pre_call(stream)

        for texref in texrefs:
            func.param_set_texref(texref)

        post_handlers = [handler
                for handler in handlers
                if hasattr(handler, "post_call")]

        if stream is None:
            if time_kernel:
                Context.synchronize()

                from time import time
                start_time = time()
            func.launch_grid(*grid)
            if post_handlers or time_kernel:
                Context.synchronize()

                if time_kernel:
                    run_time = time()-start_time

                for handler in post_handlers:
                    handler.post_call(stream)

                if time_kernel:
                    return run_time
        else:
            assert not time_kernel, "Can't time the kernel on an asynchronous invocation"
            func.launch_grid_async(grid[0], grid[1], stream)

            if post_handlers:
                for handler in post_handlers:
                    handler.post_call(stream)

    def function_prepare(func, arg_types, block, shared=None, texrefs=[]):
        func.set_block_shape(*block)

        if shared is not None:
            func.set_shared_size(shared)

        func.texrefs = texrefs

        try:
            import numpy
        except ImportError:
            numpy = None

        func.arg_format = ""
        param_size = 0

        for i, arg_type in enumerate(arg_types):
            if isinstance(arg_type, type) and numpy is not None and numpy.number in arg_type.__mro__:
                func.arg_format += numpy.dtype(arg_type).char
            elif isinstance(arg_type, str):
                func.arg_format += arg_type
            else:
                func.arg_format += numpy.dtype(numpy.intp).char

        from struct import calcsize
        func.param_set_size(calcsize(func.arg_format))

        return func

    def function_prepared_call(func, grid, *args):
        from struct import pack
        func.param_setv(0, pack(func.arg_format, *args))

        for texref in func.texrefs:
            func.param_set_texref(texref)

        func.launch_grid(*grid)

    def function_prepared_timed_call(func, grid, *args):
        from struct import pack
        func.param_setv(0, pack(func.arg_format, *args))

        for texref in func.texrefs:
            func.param_set_texref(texref)

        start = Event()
        end = Event()
        
        start.record()
        func.launch_grid(*grid)
        end.record()

        def get_call_time():
            end.synchronize()
            return end.time_since(start)*1e-3

        return get_call_time

    def function_prepared_async_call(func, grid, stream, *args):
        from struct import pack
        func.param_setv(0, pack(func.arg_format, *args))
        for texref in func.texrefs:
            func.param_set_texref(texref)

        if stream is None:
            func.launch_grid(*grid)
        else:
            grid_x, grid_y = grid
            func.launch_grid_async(grid_x, grid_y, stream)

    Device.get_attributes = device_get_attributes
    Function.param_set = function_param_set
    Function.__call__ = function_call
    Function.prepare = function_prepare
    Function.prepared_call = function_prepared_call
    Function.prepared_timed_call = function_prepared_timed_call
    Function.prepared_async_call = function_prepared_async_call




_add_functionality()




def pagelocked_zeros(shape, dtype, order="C"):
    result = pagelocked_empty(shape, dtype, order)
    result.fill(0)
    return result




def pagelocked_empty_like(array):
    if array.flags.c_contiguous:
        order = "C"
    elif array.flags.f_contiguous:
        order = "F"
    else:
        raise ValueError, "could not detect array order"

    return pagelocked_empty(array.shape, array.dtype, order)




def pagelocked_zeros_like(array):
    result = pagelocked_empty_like(array)
    result.fill(0)
    return result




def mem_alloc_like(ary):
    return mem_alloc(ary.nbytes)




def to_device(bf_obj):
    bf = buffer(bf_obj)
    result = mem_alloc(len(bf))
    memcpy_htod(result, bf)
    return result




def dtype_to_array_format(dtype):
    import numpy

    if dtype == numpy.uint8:
        return array_format.UNSIGNED_INT8
    elif dtype == numpy.uint16:
        return array_format.UNSIGNED_INT16
    elif dtype == numpy.uint32:
        return array_format.UNSIGNED_INT32
    elif dtype == numpy.int8:
        return array_format.SIGNED_INT8
    elif dtype == numpy.int16:
        return array_format.SIGNED_INT16
    elif dtype == numpy.int32:
        return array_format.SIGNED_INT32
    elif dtype == numpy.float32:
        return array_format.FLOAT
    else:
        raise TypeError(
                "cannot convert dtype '%s' to array format" 
                % dtype)




def matrix_to_array(matrix, order):
    import numpy

    if order.upper() == "C":
        h, w = matrix.shape
        stride = 0
    elif order.upper() == "F":
        w, h = matrix.shape
        stride = -1
    else: 
        raise LogicError, "order must be either F or C"

    matrix = numpy.asarray(matrix, order=order)
    descr = ArrayDescriptor()

    descr.width = w
    descr.height = h
    descr.format = dtype_to_array_format(matrix.dtype)
    descr.num_channels = 1

    ary = Array(descr)

    copy = Memcpy2D()
    copy.set_src_host(matrix)
    copy.set_dst_array(ary)
    copy.width_in_bytes = copy.src_pitch = copy.dst_pitch = \
            matrix.strides[stride]
    copy.height = h
    copy(aligned=True)

    return ary




def make_multichannel_2d_array(ndarray, order):
    """Channel count has to be the first dimension of the C{ndarray}."""

    import numpy
    ndarray = numpy.asarray(ndarray, order="F")
    descr = ArrayDescriptor()

    if order.upper() == "C":
        h, w, num_channels = ndarray.shape
        stride = 0
    elif order.upper() == "F":
        num_channels, w, h = ndarray.shape
        stride = 2
    else: 
        raise LogicError, "order must be either F or C"

    descr.width = w
    descr.height = h
    descr.format = dtype_to_array_format(ndarray.dtype)
    descr.num_channels = num_channels

    ary = Array(descr)

    copy = Memcpy2D()
    copy.set_src_host(ndarray)
    copy.set_dst_array(ary)
    copy.width_in_bytes = copy.src_pitch = copy.dst_pitch = \
            ndarray.strides[stride]
    copy.height = h
    copy(aligned=True)

    return ary




def bind_array_to_texref(ary, texref):
    texref.set_array(ary)
    texref.set_address_mode(0, address_mode.CLAMP)
    texref.set_address_mode(1, address_mode.CLAMP)
    texref.set_filter_mode(filter_mode.POINT)
    assert texref.get_flags() == 0




def matrix_to_texref(matrix, texref, order):
    bind_array_to_texref(matrix_to_array(matrix, order), texref)




def from_device(devptr, shape, dtype, order="C"):
    import numpy
    result = numpy.empty(shape, dtype, order)
    memcpy_dtoh(result, devptr)
    return result




def from_device_like(devptr, other_ary):
    import numpy
    result = numpy.empty_like(other_ary)
    memcpy_dtoh(result, devptr)
    return result





@memoize
def _get_nvcc_version(nvcc):
    from subprocess import Popen, PIPE
    try:
        return Popen([nvcc, "--version"], stdout=PIPE).communicate()[0]
    except OSError, e:
        raise OSError, "%s was not found (is it on the PATH?) [%s]" % (
                nvcc, str(e))




def _do_compile(source, options, keep, nvcc, cache_dir):
    from os.path import join

    if cache_dir:
        import md5
        checksum = md5.new()

        checksum.update(source)
        for option in options: 
            checksum.update(option)
        checksum.update(_get_nvcc_version(nvcc))

        cache_file = checksum.hexdigest()
        cache_path = join(cache_dir, cache_file + ".cubin")

        try:
            return open(cache_path, "r").read()
        except:
            pass

    from tempfile import mkdtemp
    file_dir = mkdtemp()
    file_root = "kernel"

    cu_file_name = file_root + ".cu"
    cu_file_path = join(file_dir, cu_file_name)

    outf = open(cu_file_path, "w")
    outf.write(str(source))
    outf.close()

    if keep:
        options = options[:]
        options.append("--keep")

        print "*** compiler output in %s" % file_dir

    from subprocess import call
    try:
        result = call([nvcc, "--cubin"]
                + options
                + [cu_file_name],
            cwd=file_dir)
    except OSError, e:
        raise OSError, "%s was not found (is it on the PATH?) [%s]" % (
                nvcc, str(e))

    if result != 0:
        raise CompileError, "nvcc compilation of %s failed" % cu_file_path

    cubin = open(join(file_dir, file_root + ".cubin"), "r").read()

    if cache_dir:
        outf = open(cache_path, "w")
        outf.write(cubin)
        outf.close()

    if not keep:
        from os import listdir, unlink, rmdir
        for name in listdir(file_dir):
            unlink(join(file_dir, name))
        rmdir(file_dir)

    return cubin





class SourceModule(object):
    def __init__(self, source, nvcc="nvcc",
            options=[], keep=False,
            no_extern_c=False, arch=None, code=None,
            cache_dir=None):

        if not no_extern_c:
            source = 'extern "C" {\n%s\n}\n' % source

        options = options[:]
        if arch is None:
            try:
                arch = "sm_%d%d" % Context.get_device().compute_capability()
            except RuntimeError:
                pass

        if cache_dir is None:
            from os.path import expanduser, join, exists
            import os
            from tempfile import gettempdir
            cache_dir = join(gettempdir(), 
                    "pycuda-compiler-cache-v1-uid%s" % os.getuid())

            if not exists(cache_dir):
                from os import mkdir
                mkdir(cache_dir)

        options = options[:]
        if arch is not None:
            options.extend(["-arch", arch])

        if code is not None:
            options.extend(["-code", code])

        cubin = _do_compile(source, options, keep, nvcc, cache_dir)

        def failsafe_extract(key, cubin):
            pattern = r"%s\s*=\s*([0-9]+)" % key
            import re
            match = re.search(pattern, cubin)
            if match is None:
                from warnings import warn
                warn("Reading '%s' from cubin failed--SourceModule metadata may be unavailable." % key)
                return None
            else:
                return int(match.group(1))

        self.lmem = failsafe_extract("lmem", cubin)
        self.smem = failsafe_extract("smem", cubin)
        self.registers = failsafe_extract("reg", cubin)

        self.module = module_from_buffer(cubin)

        self.get_global = self.module.get_global
        self.get_texref = self.module.get_texref

    def get_function(self, name):
        func = self.module.get_function(name)

        # FIXME: Bzzt, wrong. This should truly be per-function.
        func.lmem = self.lmem
        func.smem = self.smem
        func.registers = self.registers

        return func
