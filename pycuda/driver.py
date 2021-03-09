__copyright__ = """
Copyright 2008-2021 Andreas Kloeckner
Copyright 2021 NVIDIA Corporation
"""

import os
import numpy as np


# {{{ add cuda lib dir to Python DLL path


def _search_on_path(filenames):
    """Find file on system path."""
    # http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/52224

    from os.path import exists, abspath, join
    from os import pathsep, environ

    search_path = environ["PATH"]

    paths = search_path.split(pathsep)
    for path in paths:
        for filename in filenames:
            if exists(join(path, filename)):
                return abspath(join(path, filename))


def _add_cuda_libdir_to_dll_path():
    from os.path import join, dirname

    cuda_path = os.environ.get("CUDA_PATH")

    if cuda_path is not None:
        os.add_dll_directory(join(cuda_path, "bin"))
        return

    nvcc_path = _search_on_path(["nvcc.exe"])
    if nvcc_path is not None:
        os.add_dll_directory(dirname(nvcc_path))

    from warnings import warn

    warn(
        "Unable to discover CUDA installation directory "
        "while attempting to add it to Python's DLL path. "
        "Either set the 'CUDA_PATH' environment variable "
        "or ensure that 'nvcc.exe' is on the path."
    )


try:
    os.add_dll_directory
except AttributeError:
    # likely not on Py3.8 and Windows
    # https://github.com/inducer/pycuda/issues/213
    pass
else:
    _add_cuda_libdir_to_dll_path()

# }}}


try:
    from pycuda._driver import *  # noqa
except ImportError as e:
    if "_v2" in str(e):
        from warnings import warn

        warn(
            "Failed to import the CUDA driver interface, with an error "
            "message indicating that the version of your CUDA header "
            "does not match the version of your CUDA driver."
        )
    raise


_memoryview = memoryview
_my_bytes = bytes


try:
    ManagedAllocationOrStub = ManagedAllocation
except NameError:
    # Provide ManagedAllocationOrStub if not on CUDA 6.
    # This avoids having to do a version check in a high-traffic code path below.

    class ManagedAllocationOrStub:
        pass


CUDA_DEBUGGING = False


def set_debugging(flag=True):
    global CUDA_DEBUGGING
    CUDA_DEBUGGING = flag


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
            except Exception as e:
                print(e)
        if self.stdout:
            result += "\n[stdout:\n%s]" % self.stdout
        if self.stderr:
            result += "\n[stderr:\n%s]" % self.stderr

        return result


class ArgumentHandler:
    def __init__(self, ary):
        self.array = ary
        self.dev_alloc = None

    def get_device_alloc(self):
        if self.dev_alloc is None:
            try:
                self.dev_alloc = mem_alloc_like(self.array)
            except AttributeError:
                raise TypeError(
                    "could not determine array length of '%s': unsupported array type or not an array"
                    % type(self.array)
                )
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
            except LogicError as e:
                from warnings import warn

                warn(
                    "CUDA driver raised '%s' when querying '%s' on '%s'"
                    % (e, att_name, dev)
                )
            else:
                result[att_id] = att_value

        return result

    def device___getattr__(dev, name):
        return dev.get_attribute(getattr(device_attribute, name.upper()))

    def _build_arg_buf(args):
        handlers = []

        arg_data = []
        format = ""
        for i, arg in enumerate(args):
            if isinstance(arg, np.number):
                arg_data.append(arg)
                format += arg.dtype.char
            elif isinstance(arg, (DeviceAllocation, PooledDeviceAllocation)):
                arg_data.append(int(arg))
                format += "P"
            elif isinstance(arg, ArgumentHandler):
                handlers.append(arg)
                arg_data.append(int(arg.get_device_alloc()))
                format += "P"
            elif isinstance(arg, np.ndarray):
                if isinstance(arg.base, ManagedAllocationOrStub):
                    arg_data.append(int(arg.base))
                    format += "P"
                else:
                    arg_data.append(arg)
                    format += "%ds" % arg.nbytes
            elif isinstance(arg, np.void):
                arg_data.append(_my_bytes(_memoryview(arg)))
                format += "%ds" % arg.itemsize
            else:
                cai = getattr(arg, "__cuda_array_interface__", None)
                if cai:
                    arg_data.append(cai["data"][0])
                    format += "P"
                    continue

                try:
                    gpudata = np.uintp(arg.gpudata)
                except AttributeError:
                    raise TypeError("invalid type on parameter #%d (0-based)" % i)
                else:
                    # for gpuarrays
                    arg_data.append(int(gpudata))
                    format += "P"

        from pycuda._pvt_struct import pack

        return handlers, pack(format, *arg_data)

    # {{{ pre-CUDA 4 call interface (stateful)

    def function_param_set_pre_v4(func, *args):
        handlers = []

        handlers, buf = _build_arg_buf(args)

        func._param_setv(0, buf)
        func._param_set_size(len(buf))

        return handlers

    def function_call_pre_v4(func, *args, **kwargs):
        grid = kwargs.pop("grid", (1, 1))
        stream = kwargs.pop("stream", None)
        block = kwargs.pop("block", None)
        shared = kwargs.pop("shared", None)
        texrefs = kwargs.pop("texrefs", [])
        time_kernel = kwargs.pop("time_kernel", False)

        if kwargs:
            raise ValueError(
                "extra keyword arguments: %s" % (",".join(kwargs.keys()))
            )

        if block is None:
            raise ValueError("must specify block size")

        func._set_block_shape(*block)
        handlers = func._param_set(*args)
        if shared is not None:
            func._set_shared_size(shared)

        for handler in handlers:
            handler.pre_call(stream)

        for texref in texrefs:
            func.param_set_texref(texref)

        post_handlers = [
            handler for handler in handlers if hasattr(handler, "post_call")
        ]

        if stream is None:
            if time_kernel:
                Context.synchronize()

                from time import time

                start_time = time()
            func._launch_grid(*grid)
            if post_handlers or time_kernel:
                Context.synchronize()

                if time_kernel:
                    run_time = time() - start_time

                for handler in post_handlers:
                    handler.post_call(stream)

                if time_kernel:
                    return run_time
        else:
            assert (
                not time_kernel
            ), "Can't time the kernel on an asynchronous invocation"
            func._launch_grid_async(grid[0], grid[1], stream)

            if post_handlers:
                for handler in post_handlers:
                    handler.post_call(stream)

    def function_prepare_pre_v4(func, arg_types, block=None, shared=None, texrefs=[]):
        from warnings import warn

        if block is not None:
            warn(
                "setting the block size in Function.prepare is deprecated",
                DeprecationWarning,
                stacklevel=2,
            )
            func._set_block_shape(*block)

        if shared is not None:
            warn(
                "setting the shared memory size in Function.prepare is deprecated",
                DeprecationWarning,
                stacklevel=2,
            )
            func._set_shared_size(shared)

        func.texrefs = texrefs

        func.arg_format = ""

        for i, arg_type in enumerate(arg_types):
            if (
                isinstance(arg_type, type)
                and np is not None
                and np.number in arg_type.__mro__
            ):
                func.arg_format += np.dtype(arg_type).char
            elif isinstance(arg_type, str):
                func.arg_format += arg_type
            else:
                func.arg_format += np.dtype(np.uintp).char

        from pycuda._pvt_struct import calcsize

        func._param_set_size(calcsize(func.arg_format))

        return func

    def function_prepared_call_pre_v4(func, grid, block, *args, **kwargs):
        if isinstance(block, tuple):
            func._set_block_shape(*block)
        else:
            from warnings import warn

            warn(
                "Not passing the block size to prepared_call is deprecated as of "
                "version 2011.1.",
                DeprecationWarning,
                stacklevel=2,
            )
            args = (block,) + args

        shared_size = kwargs.pop("shared_size", None)
        if shared_size is not None:
            func._set_shared_size(shared_size)

        if kwargs:
            raise TypeError(
                "unknown keyword arguments: " + ", ".join(kwargs.keys())
            )

        from pycuda._pvt_struct import pack

        func._param_setv(0, pack(func.arg_format, *args))

        for texref in func.texrefs:
            func.param_set_texref(texref)

        func._launch_grid(*grid)

    def function_prepared_timed_call_pre_v4(func, grid, block, *args, **kwargs):
        if isinstance(block, tuple):
            func._set_block_shape(*block)
        else:
            from warnings import warn

            warn(
                "Not passing the block size to prepared_timed_call is "
                "deprecated as of version 2011.1.",
                DeprecationWarning,
                stacklevel=2,
            )
            args = (block,) + args

        shared_size = kwargs.pop("shared_size", None)
        if shared_size is not None:
            func._set_shared_size(shared_size)

        if kwargs:
            raise TypeError(
                "unknown keyword arguments: " + ", ".join(kwargs.keys())
            )

        from pycuda._pvt_struct import pack

        func._param_setv(0, pack(func.arg_format, *args))

        for texref in func.texrefs:
            func.param_set_texref(texref)

        start = Event()
        end = Event()

        start.record()
        func._launch_grid(*grid)
        end.record()

        def get_call_time():
            end.synchronize()
            return end.time_since(start) * 1e-3

        return get_call_time

    def function_prepared_async_call_pre_v4(func, grid, block, stream, *args, **kwargs):
        if isinstance(block, tuple):
            func._set_block_shape(*block)
        else:
            from warnings import warn

            warn(
                "Not passing the block size to prepared_async_call is "
                "deprecated as of version 2011.1.",
                DeprecationWarning,
                stacklevel=2,
            )
            args = (stream,) + args
            stream = block

        shared_size = kwargs.pop("shared_size", None)
        if shared_size is not None:
            func._set_shared_size(shared_size)

        if kwargs:
            raise TypeError(
                "unknown keyword arguments: " + ", ".join(kwargs.keys())
            )

        from pycuda._pvt_struct import pack

        func._param_setv(0, pack(func.arg_format, *args))

        for texref in func.texrefs:
            func.param_set_texref(texref)

        if stream is None:
            func._launch_grid(*grid)
        else:
            grid_x, grid_y = grid
            func._launch_grid_async(grid_x, grid_y, stream)

    # }}}

    # {{{ CUDA 4+ call interface (stateless)

    def function_call(func, *args, **kwargs):
        grid = kwargs.pop("grid", (1, 1))
        stream = kwargs.pop("stream", None)
        block = kwargs.pop("block", None)
        shared = kwargs.pop("shared", 0)
        texrefs = kwargs.pop("texrefs", [])
        time_kernel = kwargs.pop("time_kernel", False)

        if kwargs:
            raise ValueError(
                "extra keyword arguments: %s" % (",".join(kwargs.keys()))
            )

        if block is None:
            raise ValueError("must specify block size")

        func._set_block_shape(*block)
        handlers, arg_buf = _build_arg_buf(args)

        for handler in handlers:
            handler.pre_call(stream)

        for texref in texrefs:
            func.param_set_texref(texref)

        post_handlers = [
            handler for handler in handlers if hasattr(handler, "post_call")
        ]

        if stream is None:
            if time_kernel:
                Context.synchronize()

                from time import time

                start_time = time()

            func._launch_kernel(grid, block, arg_buf, shared, None)

            if post_handlers or time_kernel:
                Context.synchronize()

                if time_kernel:
                    run_time = time() - start_time

                for handler in post_handlers:
                    handler.post_call(stream)

                if time_kernel:
                    return run_time
        else:
            assert (
                not time_kernel
            ), "Can't time the kernel on an asynchronous invocation"
            func._launch_kernel(grid, block, arg_buf, shared, stream)

            if post_handlers:
                for handler in post_handlers:
                    handler.post_call(stream)

    def function_prepare(func, arg_types, texrefs=[]):
        func.texrefs = texrefs

        func.arg_format = ""

        for i, arg_type in enumerate(arg_types):
            if isinstance(arg_type, type) and np.number in arg_type.__mro__:
                func.arg_format += np.dtype(arg_type).char
            elif isinstance(arg_type, np.dtype):
                if arg_type.char == "V":
                    func.arg_format += "%ds" % arg_type.itemsize
                else:
                    func.arg_format += arg_type.char
            elif isinstance(arg_type, str):
                func.arg_format += arg_type
            else:
                func.arg_format += np.dtype(np.uintp).char

        return func

    def function_prepared_call(func, grid, block, *args, **kwargs):
        if isinstance(block, tuple):
            func._set_block_shape(*block)
        else:
            from warnings import warn

            warn(
                "Not passing the block size to prepared_call is deprecated as of "
                "version 2011.1.",
                DeprecationWarning,
                stacklevel=2,
            )
            args = (block,) + args

        shared_size = kwargs.pop("shared_size", 0)

        if kwargs:
            raise TypeError(
                "unknown keyword arguments: " + ", ".join(kwargs.keys())
            )

        from pycuda._pvt_struct import pack

        arg_buf = pack(func.arg_format, *args)

        for texref in func.texrefs:
            func.param_set_texref(texref)

        func._launch_kernel(grid, block, arg_buf, shared_size, None)

    def function_prepared_timed_call(func, grid, block, *args, **kwargs):
        shared_size = kwargs.pop("shared_size", 0)
        if kwargs:
            raise TypeError(
                "unknown keyword arguments: " + ", ".join(kwargs.keys())
            )

        from pycuda._pvt_struct import pack

        arg_buf = pack(func.arg_format, *args)

        for texref in func.texrefs:
            func.param_set_texref(texref)

        start = Event()
        end = Event()

        start.record()
        func._launch_kernel(grid, block, arg_buf, shared_size, None)
        end.record()

        def get_call_time():
            end.synchronize()
            return end.time_since(start) * 1e-3

        return get_call_time

    def function_prepared_async_call(func, grid, block, stream, *args, **kwargs):
        if isinstance(block, tuple):
            func._set_block_shape(*block)
        else:
            from warnings import warn

            warn(
                "Not passing the block size to prepared_async_call is "
                "deprecated as of version 2011.1.",
                DeprecationWarning,
                stacklevel=2,
            )
            args = (stream,) + args
            stream = block

        shared_size = kwargs.pop("shared_size", 0)

        if kwargs:
            raise TypeError(
                "unknown keyword arguments: " + ", ".join(kwargs.keys())
            )

        from pycuda._pvt_struct import pack

        arg_buf = pack(func.arg_format, *args)

        for texref in func.texrefs:
            func.param_set_texref(texref)

        func._launch_kernel(grid, block, arg_buf, shared_size, stream)

    # }}}

    def function___getattr__(self, name):
        if get_version() >= (2, 2):
            return self.get_attribute(getattr(function_attribute, name.upper()))
        else:
            if name == "num_regs":
                return self._hacky_registers
            elif name == "shared_size_bytes":
                return self._hacky_smem
            elif name == "local_size_bytes":
                return self._hacky_lmem
            else:
                raise AttributeError("no attribute '%s' in Function" % name)

    def mark_func_method_deprecated(func):
        def new_func(*args, **kwargs):
            from warnings import warn

            warn(
                "'%s' has been deprecated in version 2011.1. Please use "
                "the stateless launch interface instead." % func.__name__[1:],
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        try:
            from functools import update_wrapper
        except ImportError:
            pass
        else:
            try:
                update_wrapper(new_func, func)
            except Exception:
                # User won't see true signature. Oh well.
                pass

        return new_func

    Device.get_attributes = device_get_attributes
    Device.__getattr__ = device___getattr__

    if get_version() >= (4,):
        Function.__call__ = function_call
        Function.prepare = function_prepare
        Function.prepared_call = function_prepared_call
        Function.prepared_timed_call = function_prepared_timed_call
        Function.prepared_async_call = function_prepared_async_call
    else:
        Function._param_set = function_param_set_pre_v4
        Function.__call__ = function_call_pre_v4
        Function.prepare = function_prepare_pre_v4
        Function.prepared_call = function_prepared_call_pre_v4
        Function.prepared_timed_call = function_prepared_timed_call_pre_v4
        Function.prepared_async_call = function_prepared_async_call_pre_v4

        for meth_name in [
            "set_block_shape",
            "set_shared_size",
            "param_set_size",
            "param_set",
            "param_seti",
            "param_setf",
            "param_setv",
            "launch",
            "launch_grid",
            "launch_grid_async",
        ]:
            setattr(
                Function,
                meth_name,
                mark_func_method_deprecated(getattr(Function, "_" + meth_name)),
            )

    Function.__getattr__ = function___getattr__


_add_functionality()


# {{{ pagelocked numpy arrays


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
        raise ValueError("could not detect array order")

    return pagelocked_empty(array.shape, array.dtype, order, mem_flags)


def pagelocked_zeros_like(array, mem_flags=0):
    result = pagelocked_empty_like(array, mem_flags)
    result.fill(0)
    return result


# }}}


# {{{ aligned numpy arrays


def aligned_zeros(shape, dtype, order="C", alignment=4096):
    result = aligned_empty(shape, dtype, order, alignment)
    result.fill(0)
    return result


def aligned_empty_like(array, alignment=4096):
    if array.flags.c_contiguous:
        order = "C"
    elif array.flags.f_contiguous:
        order = "F"
    else:
        raise ValueError("could not detect array order")

    return aligned_empty(array.shape, array.dtype, order, alignment)


def aligned_zeros_like(array, alignment=4096):
    result = aligned_empty_like(array, alignment)
    result.fill(0)
    return result


# }}}


# {{{ managed numpy arrays (CUDA Unified Memory)


def managed_zeros(shape, dtype, order="C", mem_flags=0):
    result = managed_empty(shape, dtype, order, mem_flags)
    result.fill(0)
    return result


def managed_empty_like(array, mem_flags=0):
    if array.flags.c_contiguous:
        order = "C"
    elif array.flags.f_contiguous:
        order = "F"
    else:
        raise ValueError("could not detect array order")

    return managed_empty(array.shape, array.dtype, order, mem_flags)


def managed_zeros_like(array, mem_flags=0):
    result = managed_empty_like(array, mem_flags)
    result.fill(0)
    return result


# }}}


def mem_alloc_like(ary):
    return mem_alloc(ary.nbytes)


# {{{ array handling


def dtype_to_array_format(dtype):
    if dtype == np.uint8:
        return array_format.UNSIGNED_INT8
    elif dtype == np.uint16:
        return array_format.UNSIGNED_INT16
    elif dtype == np.uint32:
        return array_format.UNSIGNED_INT32
    elif dtype == np.int8:
        return array_format.SIGNED_INT8
    elif dtype == np.int16:
        return array_format.SIGNED_INT16
    elif dtype == np.int32:
        return array_format.SIGNED_INT32
    elif dtype == np.float32:
        return array_format.FLOAT
    else:
        raise TypeError("cannot convert dtype '%s' to array format" % dtype)


def matrix_to_array(matrix, order, allow_double_hack=False):
    if order.upper() == "C":
        h, w = matrix.shape
        stride = 0
    elif order.upper() == "F":
        w, h = matrix.shape
        stride = -1
    else:
        raise LogicError("order must be either F or C")

    matrix = np.asarray(matrix, order=order)
    descr = ArrayDescriptor()

    descr.width = w
    descr.height = h

    if matrix.dtype == np.float64 and allow_double_hack:
        descr.format = array_format.SIGNED_INT32
        descr.num_channels = 2
    else:
        descr.format = dtype_to_array_format(matrix.dtype)
        descr.num_channels = 1

    ary = Array(descr)

    copy = Memcpy2D()
    copy.set_src_host(matrix)
    copy.set_dst_array(ary)
    copy.width_in_bytes = copy.src_pitch = copy.dst_pitch = matrix.strides[stride]
    copy.height = h
    copy(aligned=True)

    return ary


def np_to_array(nparray, order, allowSurfaceBind=False):  # noqa: N803
    case = order in ["C", "F"]
    if not case:
        raise LogicError("order must be either F or C")

    dimension = len(nparray.shape)
    if dimension == 2:
        if order == "C":
            stride = 0
        if order == "F":
            stride = -1
        h, w = nparray.shape
        d = 1
        if allowSurfaceBind:
            descrArr = ArrayDescriptor3D()
            descrArr.width = w
            descrArr.height = h
            descrArr.depth = d
        else:
            descrArr = ArrayDescriptor()
            descrArr.width = w
            descrArr.height = h
    elif dimension == 3:
        if order == "C":
            stride = 1
        if order == "F":
            stride = 1
        d, h, w = nparray.shape
        descrArr = ArrayDescriptor3D()
        descrArr.width = w
        descrArr.height = h
        descrArr.depth = d
    else:
        raise LogicError(
            "CUDArrays dimensions 2 or 3 supported in CUDA at the moment ... "
        )

    if nparray.dtype == np.complex64:
        descrArr.format = (
            array_format.SIGNED_INT32
        )  # Reading data as int2 (hi=re,lo=im) structure
        descrArr.num_channels = 2
    elif nparray.dtype == np.float64:
        descrArr.format = (
            array_format.SIGNED_INT32
        )  # Reading data as int2 (hi,lo) structure
        descrArr.num_channels = 2
    elif nparray.dtype == np.complex128:
        descrArr.format = (
            array_format.SIGNED_INT32
        )  # Reading data as int4 (re=(hi,lo),im=(hi,lo)) structure
        descrArr.num_channels = 4
    else:
        descrArr.format = dtype_to_array_format(nparray.dtype)
        descrArr.num_channels = 1

    if allowSurfaceBind:
        if dimension == 2:
            descrArr.flags |= array3d_flags.ARRAY3D_LAYERED
        descrArr.flags |= array3d_flags.SURFACE_LDST

    cudaArray = Array(descrArr)
    if allowSurfaceBind or dimension == 3:
        copy3D = Memcpy3D()
        copy3D.set_src_host(nparray)
        copy3D.set_dst_array(cudaArray)
        copy3D.width_in_bytes = copy3D.src_pitch = nparray.strides[stride]
        copy3D.src_height = copy3D.height = h
        copy3D.depth = d
        copy3D()
        return cudaArray
    else:
        copy2D = Memcpy2D()
        copy2D.set_src_host(nparray)
        copy2D.set_dst_array(cudaArray)
        copy2D.width_in_bytes = copy2D.src_pitch = nparray.strides[stride]
        copy2D.src_height = copy2D.height = h
        copy2D(aligned=True)
        return cudaArray


def gpuarray_to_array(gpuarray, order, allowSurfaceBind=False):  # noqa: N803
    case = order in ["C", "F"]
    if not case:
        raise LogicError("order must be either F or C")

    dimension = len(gpuarray.shape)
    if dimension == 2:
        if order == "C":
            stride = 0
        if order == "F":
            stride = -1
        h, w = gpuarray.shape
        d = 1
        if allowSurfaceBind:
            descrArr = ArrayDescriptor3D()
            descrArr.width = int(w)
            descrArr.height = int(h)
            descrArr.depth = int(d)
        else:
            descrArr = ArrayDescriptor()
            descrArr.width = int(w)
            descrArr.height = int(h)
    elif dimension == 3:
        if order == "C":
            stride = 1
        if order == "F":
            stride = 1
        d, h, w = gpuarray.shape
        descrArr = ArrayDescriptor3D()
        descrArr.width = int(w)
        descrArr.height = int(h)
        descrArr.depth = int(d)
    else:
        raise LogicError(
            "CUDArray dimensions 2 and 3 supported in CUDA at the moment ... "
        )

    if gpuarray.dtype == np.complex64:
        descrArr.format = (
            array_format.SIGNED_INT32
        )  # Reading data as int2 (hi=re,lo=im) structure
        descrArr.num_channels = 2
    elif gpuarray.dtype == np.float64:
        descrArr.format = (
            array_format.SIGNED_INT32
        )  # Reading data as int2 (hi,lo) structure
        descrArr.num_channels = 2
    elif gpuarray.dtype == np.complex128:
        descrArr.format = (
            array_format.SIGNED_INT32
        )  # Reading data as int4 (re=(hi,lo),im=(hi,lo)) structure
        descrArr.num_channels = 4
    else:
        descrArr.format = dtype_to_array_format(gpuarray.dtype)
        descrArr.num_channels = 1

    if allowSurfaceBind:
        if dimension == 2:
            descrArr.flags |= array3d_flags.ARRAY3D_LAYERED
        descrArr.flags |= array3d_flags.SURFACE_LDST

    cudaArray = Array(descrArr)
    if allowSurfaceBind or dimension == 3:
        copy3D = Memcpy3D()
        copy3D.set_src_device(gpuarray.ptr)
        copy3D.set_dst_array(cudaArray)
        copy3D.width_in_bytes = copy3D.src_pitch = gpuarray.strides[stride]
        copy3D.src_height = copy3D.height = int(h)
        copy3D.depth = int(d)
        copy3D()
        return cudaArray
    else:
        copy2D = Memcpy2D()
        copy2D.set_src_device(gpuarray.ptr)
        copy2D.set_dst_array(cudaArray)
        copy2D.width_in_bytes = copy2D.src_pitch = gpuarray.strides[stride]
        copy2D.src_height = copy2D.height = int(h)
        copy2D(aligned=True)
        return cudaArray


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
        raise LogicError("order must be either F or C")

    descr.width = w
    descr.height = h
    descr.format = dtype_to_array_format(ndarray.dtype)
    descr.num_channels = num_channels

    ary = Array(descr)

    copy = Memcpy2D()
    copy.set_src_host(ndarray)
    copy.set_dst_array(ary)
    copy.width_in_bytes = copy.src_pitch = copy.dst_pitch = ndarray.strides[stride]
    copy.height = h
    copy(aligned=True)

    return ary


def bind_array_to_texref(ary, texref):
    texref.set_array(ary)
    texref.set_address_mode(0, address_mode.CLAMP)
    texref.set_address_mode(1, address_mode.CLAMP)
    texref.set_filter_mode(filter_mode.POINT)


# }}}


def matrix_to_texref(matrix, texref, order):
    bind_array_to_texref(matrix_to_array(matrix, order), texref)


# {{{ device copies


def to_device(bf_obj):
    import sys

    if sys.version_info >= (2, 7):
        bf = memoryview(bf_obj).tobytes()
    else:
        bf = buffer(bf_obj)
    result = mem_alloc(len(bf))
    memcpy_htod(result, bf)
    return result


def from_device(devptr, shape, dtype, order="C"):
    result = np.empty(shape, dtype, order)
    memcpy_dtoh(result, devptr)
    return result


def from_device_like(devptr, other_ary):
    result = np.empty_like(other_ary)
    memcpy_dtoh(result, devptr)
    return result


# }}}

# vim: fdm=marker
