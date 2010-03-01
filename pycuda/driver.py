from _driver import *




class CompileError(Error):
    def __init__(self, msg, command_line, stdout=None, stderr=None):
        self.msg = msg
        self.command_line = command_line
        self.stdout = stdout
        self.stderr = stderr

    def __str__(self):
        result = self.msg
        if self.command_line:
            try:
                result += "\n[command: %s]" % (" ".join(self.command_line))
            except Exception, e:
                print e
        if self.stdout:
            result += "\n[stdout:\n%s]" % self.stdout
        if self.stderr:
            result += "\n[stderr:\n%s]" % self.stderr

        return result




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
        if stream is not None:
            memcpy_htod(self.get_device_alloc(), self.array)
        else:
            memcpy_htod(self.get_device_alloc(), self.array)

class Out(ArgumentHandler):
    def post_call(self, stream):
        if stream is not None:
            memcpy_dtoh(self.array, self.get_device_alloc())
        else:
            memcpy_dtoh(self.array, self.get_device_alloc())

class InOut(In, Out):
    pass





def _add_functionality():
    def device_get_attributes(dev):
        result = {}

        for att_name in dir(device_attribute):
            if not att_name[0].isupper():
                continue

            att_id = getattr(device_attribute, att_name)

            try:
                att_value = dev.get_attribute(att_id)
            except LogicError, e:
                from warnings import warn
                warn("CUDA driver raised '%s' when querying '%s' on '%s'"
                        % (e, att_name, dev))
            else:
                result[att_id] = att_value

        return result

    def device___getattr__(dev, name):
        return dev.get_attribute(getattr(device_attribute, name.upper()))

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

        from pycuda._pvt_struct import pack
        buf = pack(format, *arg_data)

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

        from pycuda._pvt_struct import calcsize
        func.param_set_size(calcsize(func.arg_format))

        return func

    def function_prepared_call(func, grid, *args):
        from pycuda._pvt_struct import pack
        func.param_setv(0, pack(func.arg_format, *args))

        for texref in func.texrefs:
            func.param_set_texref(texref)

        func.launch_grid(*grid)

    def function_prepared_timed_call(func, grid, *args):
        from pycuda._pvt_struct import pack
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
        from pycuda._pvt_struct import pack
        func.param_setv(0, pack(func.arg_format, *args))

        for texref in func.texrefs:
            func.param_set_texref(texref)

        if stream is None:
            func.launch_grid(*grid)
        else:
            grid_x, grid_y = grid
            func.launch_grid_async(grid_x, grid_y, stream)

    def function___getattr__(self, name):
        if get_version() >= (2,2):
            return self.get_attribute(getattr(function_attribute, name.upper()))
        else:
            if name == "num_regs": return self._hacky_registers
            elif name == "shared_size_bytes": return self._hacky_smem
            elif name == "local_size_bytes": return self._hacky_lmem
            else:
                raise AttributeError("no attribute '%s' in Function" % name)

    Device.get_attributes = device_get_attributes
    Device.__getattr__ = device___getattr__
    Function.param_set = function_param_set
    Function.__call__ = function_call
    Function.prepare = function_prepare
    Function.prepared_call = function_prepared_call
    Function.prepared_timed_call = function_prepared_timed_call
    Function.prepared_async_call = function_prepared_async_call
    Function.__getattr__ = function___getattr__




_add_functionality()




def pagelocked_zeros(shape, dtype, order="C", mem_flags=0):
    result = pagelocked_empty(shape, dtype, order, mem_flags)
    result.fill(0)
    return result




def pagelocked_empty_like(array, mem_flags=0):
    if array.flags.c_contiguous:
        order = "C"
    elif array.flags.f_contiguous:
        order = "F"
    else:
        raise ValueError, "could not detect array order"

    return pagelocked_empty(array.shape, array.dtype, order, mem_flags)




def pagelocked_zeros_like(array, mem_flags=0):
    result = pagelocked_empty_like(array, mem_flags)
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




def matrix_to_array(matrix, order, allow_double_hack=False):
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

    if matrix.dtype == numpy.float64 and allow_double_hack:
        descr.format = array_format.SIGNED_INT32
        descr.num_channels = 2
    else:
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
