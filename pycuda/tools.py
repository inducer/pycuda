"""Miscallenous helper functionality."""

__copyright__ = "Copyright (C) 2008 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

import pycuda.driver as cuda
import pycuda._driver as _drv
import numpy as np


from pycuda.compyte.dtypes import (  # noqa: F401
    register_dtype,
    get_or_register_dtype,
    _fill_dtype_registry,
    dtype_to_ctype as base_dtype_to_ctype,
)

bitlog2 = _drv.bitlog2
DeviceMemoryPool = _drv.DeviceMemoryPool
PageLockedMemoryPool = _drv.PageLockedMemoryPool
PageLockedAllocator = _drv.PageLockedAllocator

_fill_dtype_registry(respect_windows=True)
get_or_register_dtype("pycuda::complex<float>", np.complex64)
get_or_register_dtype("pycuda::complex<double>", np.complex128)


# {{{ debug memory pool


class DebugMemoryPool(DeviceMemoryPool):
    def __init__(self, interactive=True, logfile=None):
        DeviceMemoryPool.__init__(self)
        self.last_free, _ = cuda.mem_get_info()
        self.interactive = interactive

        if logfile is None:
            import sys

            logfile = sys.stdout

        self.logfile = logfile

        from weakref import WeakKeyDictionary

        self.blocks = WeakKeyDictionary()

        if interactive:
            from pytools.diskdict import DiskDict

            self.stacktrace_mnemonics = DiskDict("pycuda-stacktrace-mnemonics")

    def allocate(self, size):
        from traceback import extract_stack

        stack = tuple(frm[2] for frm in extract_stack())
        description = self.describe(stack, size)

        histogram = {}
        for bsize, descr in self.blocks.values():
            histogram[bsize, descr] = histogram.get((bsize, descr), 0) + 1

        from pytools import common_prefix

        cpfx = common_prefix(descr for bsize, descr in histogram)

        print(
            "\n  Allocation of size %d occurring "
            "(mem: last_free:%d, free: %d, total:%d) (pool: held:%d, active:%d):"
            "\n      at: %s"
            % (
                (size, self.last_free)
                + cuda.mem_get_info()
                + (self.held_blocks, self.active_blocks, description)
            ),
            file=self.logfile,
        )

        hist_items = sorted(list(histogram.items()))
        for (bsize, descr), count in hist_items:
            print(
                "  %s (%d bytes): %dx" % (descr[len(cpfx):], bsize, count),
                file=self.logfile,
            )

        if self.interactive:
            input("  [Enter]")

        result = DeviceMemoryPool.allocate(self, size)
        self.blocks[result] = size, description
        self.last_free, _ = cuda.mem_get_info()
        return result

    def describe(self, stack, size):
        if not self.interactive:
            return "|".join(stack)
        else:
            try:
                return self.stacktrace_mnemonics[stack, size]
            except KeyError:
                print(size, stack)
                while True:
                    mnemonic = input("Enter mnemonic or [Enter] for more info:")
                    if mnemonic == "":
                        from traceback import print_stack

                        print_stack()
                    else:
                        break
                self.stacktrace_mnemonics[stack, size] = mnemonic
                return mnemonic


# }}}


# {{{ default device/context


def get_default_device(default=0):
    from warnings import warn

    warn(
        "get_default_device() is deprecated; " "use make_default_context() instead",
        DeprecationWarning,
    )

    from pycuda.driver import Device
    import os

    dev = os.environ.get("CUDA_DEVICE")

    if dev is None:
        try:
            dev = (
                open(os.path.join(os.path.expanduser("~"), ".cuda_device"))
                .read()
                .strip()
            )
        except Exception:
            pass

    if dev is None:
        dev = default

    try:
        dev = int(dev)
    except TypeError:
        raise TypeError(
            "CUDA device number (CUDA_DEVICE or ~/.cuda-device) " "must be an integer"
        )

    return Device(dev)


def make_default_context(ctx_maker=None):
    if ctx_maker is None:

        def ctx_maker(dev):
            return dev.make_context()

    ndevices = cuda.Device.count()
    if ndevices == 0:
        raise RuntimeError(
            "No CUDA enabled device found. " "Please check your installation."
        )

    # Is CUDA_DEVICE set?
    import os

    devn = os.environ.get("CUDA_DEVICE")

    # Is $HOME/.cuda_device set ?
    if devn is None:
        try:
            homedir = os.environ.get("HOME")
            assert homedir is not None
            devn = open(os.path.join(homedir, ".cuda_device")).read().strip()
        except Exception:
            pass

    # If either CUDA_DEVICE or $HOME/.cuda_device is set, try to use it
    if devn is not None:
        try:
            devn = int(devn)
        except TypeError:
            raise TypeError(
                "CUDA device number (CUDA_DEVICE or ~/.cuda_device)"
                " must be an integer"
            )

        dev = cuda.Device(devn)
        return ctx_maker(dev)

    # Otherwise, try to use any available device
    else:
        for devn in range(ndevices):
            dev = cuda.Device(devn)
            try:
                return ctx_maker(dev)
            except cuda.Error:
                pass

        raise RuntimeError(
            "make_default_context() wasn't able to create a context "
            "on any of the %d detected devices" % ndevices
        )


# }}}


# {{{ rounding helpers


def _exact_div(dividend, divisor):
    quot, rem = divmod(dividend, divisor)
    assert rem == 0
    return quot


def _int_ceiling(value, multiple_of=1):
    """Round C{value} up to be a C{multiple_of} something."""
    # Mimicks the Excel "floor" function (for code stolen from occupancy calculator)

    from math import ceil

    return int(ceil(value / multiple_of)) * multiple_of


def _int_floor(value, multiple_of=1):
    """Round C{value} down to be a C{multiple_of} something."""
    # Mimicks the Excel "floor" function (for code stolen from occupancy calculator)

    from math import floor

    return int(floor(value / multiple_of)) * multiple_of


# }}}


# {{{ device data


class DeviceData:
    def __init__(self, dev=None):
        import pycuda.driver as drv

        if dev is None:
            dev = cuda.Context.get_device()

        self.max_threads = dev.get_attribute(drv.device_attribute.MAX_THREADS_PER_BLOCK)
        self.warp_size = dev.get_attribute(drv.device_attribute.WARP_SIZE)

        if dev.compute_capability() >= (3, 0):
            self.warps_per_mp = 64
        elif dev.compute_capability() >= (2, 0):
            self.warps_per_mp = 48
        elif dev.compute_capability() >= (1, 2):
            self.warps_per_mp = 32
        else:
            self.warps_per_mp = 24

        self.thread_blocks_per_mp = 8
        self.registers = dev.get_attribute(drv.device_attribute.MAX_REGISTERS_PER_BLOCK)
        self.shared_memory = dev.get_attribute(
            drv.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK
        )

        if dev.compute_capability() >= (2, 0):
            self.smem_alloc_granularity = 128
            self.smem_granularity = 32
        else:
            self.smem_alloc_granularity = 512
            self.smem_granularity = 16

        if dev.compute_capability() >= (2, 0):
            self.register_allocation_unit = "warp"
        else:
            self.register_allocation_unit = "block"

    def align(self, bytes, word_size=4):
        return _int_ceiling(bytes, self.align_bytes(word_size))

    def align_dtype(self, elements, dtype_size):
        return _int_ceiling(elements, self.align_words(dtype_size))

    def align_words(self, word_size):
        return _exact_div(self.align_bytes(word_size), word_size)

    def align_bytes(self, word_size=4):
        if word_size == 4:
            return 64
        elif word_size == 8:
            return 128
        elif word_size == 16:
            return 128
        else:
            raise ValueError("no alignment possible for fetches of size %d" % word_size)

    def coalesce(self, thread_count):
        return _int_ceiling(thread_count, 16)

    @staticmethod
    def make_valid_tex_channel_count(size):
        valid_sizes = [1, 2, 4]
        for vs in valid_sizes:
            if size <= vs:
                return vs

        raise ValueError("could not enlarge argument to valid channel count")


# }}}

# {{{ occupancy


class OccupancyRecord:
    def __init__(self, devdata, threads, shared_mem=0, registers=0):
        if threads > devdata.max_threads:
            raise ValueError("too many threads")

        # copied literally from occupancy calculator
        alloc_warps = _int_ceiling(threads / devdata.warp_size)
        alloc_smem = _int_ceiling(shared_mem, devdata.smem_alloc_granularity)
        if devdata.register_allocation_unit == "warp":
            alloc_regs = alloc_warps * 32 * registers
        elif devdata.register_allocation_unit == "block":
            alloc_regs = _int_ceiling(alloc_warps * 2, 4) * 16 * registers
        else:
            raise ValueError(
                "Improper register allocation unit:" + devdata.register_allocation_unit
            )

        if alloc_regs > devdata.registers:
            raise ValueError("too many registers")

        if alloc_smem > devdata.shared_memory:
            raise ValueError("too much smem")

        self.tb_per_mp_limits = [
            (devdata.thread_blocks_per_mp, "device"),
            (_int_floor(devdata.warps_per_mp / alloc_warps), "warps"),
        ]
        if registers > 0:
            self.tb_per_mp_limits.append(
                (_int_floor(devdata.registers / alloc_regs), "regs")
            )
        if shared_mem > 0:
            self.tb_per_mp_limits.append(
                (_int_floor(devdata.shared_memory / alloc_smem), "smem")
            )

        self.tb_per_mp, self.limited_by = min(self.tb_per_mp_limits)

        self.warps_per_mp = self.tb_per_mp * alloc_warps
        self.occupancy = self.warps_per_mp / devdata.warps_per_mp


# }}}

# {{{ C types <-> dtypes


class Argument:
    def __init__(self, dtype, name):
        self.dtype = np.dtype(dtype)
        self.name = name

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r}, {self.dtype})"


def dtype_to_ctype(dtype, with_fp_tex_hack=False):
    if dtype is None:
        raise ValueError("dtype may not be None")

    dtype = np.dtype(dtype)
    if with_fp_tex_hack:
        if dtype == np.float32:
            return "fp_tex_float"
        elif dtype == np.float64:
            return "fp_tex_double"
        elif dtype == np.complex64:
            return "fp_tex_cfloat"
        elif dtype == np.complex128:
            return "fp_tex_cdouble"

    return base_dtype_to_ctype(dtype)


class VectorArg(Argument):
    def declarator(self):
        return "{} *{}".format(dtype_to_ctype(self.dtype), self.name)

    struct_char = "P"


class ScalarArg(Argument):
    def declarator(self):
        return "{} {}".format(dtype_to_ctype(self.dtype), self.name)

    @property
    def struct_char(self):
        result = self.dtype.char
        if result == "V":
            result = "%ds" % self.dtype.itemsize

        return result


def parse_c_arg(c_arg):
    from pycuda.compyte.dtypes import parse_c_arg_backend

    return parse_c_arg_backend(c_arg, ScalarArg, VectorArg)


def get_arg_type(c_arg):
    return parse_c_arg(c_arg).struct_char


# }}}

# {{{ context-dep memoization

context_dependent_memoized_functions = []


def context_dependent_memoize(func):
    def wrapper(*args, **kwargs):
        if kwargs:
            cache_key = (args, frozenset(kwargs.items()))
        else:
            cache_key = (args,)

        try:
            ctx_dict = func._pycuda_ctx_dep_memoize_dic
        except AttributeError:
            # FIXME: This may keep contexts alive longer than desired.
            # But I guess since the memory in them is freed, who cares.
            ctx_dict = func._pycuda_ctx_dep_memoize_dic = {}

        cur_ctx = cuda.Context.get_current()

        try:
            return ctx_dict[cur_ctx][cache_key]
        except KeyError:
            context_dependent_memoized_functions.append(func)
            arg_dict = ctx_dict.setdefault(cur_ctx, {})
            result = func(*args, **kwargs)
            arg_dict[cache_key] = result
            return result

    from functools import update_wrapper
    update_wrapper(wrapper, func)
    return wrapper


def clear_context_caches():
    for func in context_dependent_memoized_functions:
        try:
            ctx_dict = func._pycuda_ctx_dep_memoize_dic
        except AttributeError:
            pass
        else:
            ctx_dict.clear()


# }}}

# {{{ py.test interaction


def mark_cuda_test(inner_f):
    def f(*args, **kwargs):
        import pycuda.driver

        # appears to be idempotent, i.e. no harm in calling it more than once
        pycuda.driver.init()

        ctx = make_default_context()
        try:
            assert isinstance(ctx.get_device().name(), str)
            assert isinstance(ctx.get_device().compute_capability(), tuple)
            assert isinstance(ctx.get_device().get_attributes(), dict)
            inner_f(*args, **kwargs)
        finally:
            ctx.pop()

            from pycuda.tools import clear_context_caches

            clear_context_caches()

            from gc import collect

            collect()

    try:
        from py.test import mark as mark_test
    except ImportError:
        return f

    return mark_test.cuda(f)


# }}}


# vim: foldmethod=marker
