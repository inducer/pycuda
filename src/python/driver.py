from _driver import *




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
            elif isinstance(arg, DeviceAllocation):
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
                raise TypeError("invalid type on parameter #%d (0-based)" % i)

        import struct
        buf = struct.pack(format, *arg_data)
        func.param_setv(0, buf)
        func.param_set_size(len(buf))

        return handlers

    def function_call(func, *args, **kwargs):
        grid = kwargs.get("grid", (1,1))
        stream = kwargs.get("stream")
        block = kwargs.get("block")
        shared = kwargs.get("shared")
        texrefs = kwargs.get("texrefs", [])

        if block is not None:
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
            func.launch_grid(*grid)
            if post_handlers:
                Context.synchronize()
                for handler in post_handlers:
                    handler.post_call(stream)
        else:
            func.launch_grid_async(grid[0], grid[1], stream)

            if post_handlers:
                for handler in post_handlers:
                    handler.post_call(stream)


    Device.get_attributes = device_get_attributes
    Function.param_set = function_param_set
    Function.__call__ = function_call




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




def matrix_to_texref(matrix, texref):
    import numpy
    matrix = numpy.asarray(matrix, dtype=numpy.float32, order="F")
    descr = ArrayDescriptor()
    h, w = matrix.shape
    descr.width = h # matrices are row-first
    descr.height = w # matrices are row-first
    descr.format = array_format.FLOAT
    descr.num_channels = 1

    ary = Array(descr)

    copy = Memcpy2D()
    copy.set_src_host(matrix)
    copy.set_dst_array(ary)
    copy.width_in_bytes = copy.src_pitch = copy.dst_pitch = matrix.strides[1]
    copy.height = w
    copy(aligned=True)

    texref.set_array(ary)
    texref.set_address_mode(0, address_mode.CLAMP)
    texref.set_address_mode(1, address_mode.CLAMP)
    texref.set_filter_mode(filter_mode.POINT)
    assert texref.get_flags() == 0
    assert texref.get_format() == (array_format.FLOAT, 1)




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




class SourceModule(object):
    def __init__(self, source, options=[], keep=False, no_extern_c=False):
        from tempfile import mkdtemp
        tempdir = mkdtemp()

        from os.path import join
        outf = open(join(tempdir, "kernel.cu"), "w")
        if not no_extern_c:
            outf.write('extern "C" {\n')
        outf.write(source)
        if not no_extern_c:
            outf.write('}\n')
        outf.write("\n")
        outf.close()

        if keep:
            options = options[:]
            options.append("--keep")
            print "*** compiler output in %s" % tempdir

        from subprocess import call
        result = call(["nvcc", "--cubin"] 
                + options
                + ["kernel.cu"],
            cwd=tempdir)
        if result != 0:
            raise RuntimeError, "module compilation failed"

        data = open(join(tempdir, "kernel.cubin"), "r").read()
        self.module = module_from_buffer(data)

        if not keep:
            from os import listdir, unlink, rmdir
            for name in listdir(tempdir):
                unlink(join(tempdir, name))
            rmdir(tempdir)

        self.get_function = self.module.get_function
        self.get_global = self.module.get_global
        self.get_texref = self.module.get_texref
