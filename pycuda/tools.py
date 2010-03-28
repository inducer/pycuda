"""Miscallenous helper functionality."""

from __future__ import division

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
from pytools import memoize, decorator
import pycuda._driver as _drv
import numpy




bitlog2 = _drv.bitlog2
DeviceMemoryPool = _drv.DeviceMemoryPool
PageLockedMemoryPool = _drv.PageLockedMemoryPool



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
        for bsize, descr in self.blocks.itervalues():
            histogram[bsize, descr] = histogram.get((bsize, descr), 0) + 1

        from pytools import common_prefix
        cpfx = common_prefix(descr for bsize, descr in histogram)

        print >> self.logfile, \
                "\n  Allocation of size %d occurring " \
                "(mem: last_free:%d, free: %d, total:%d) (pool: held:%d, active:%d):" \
                "\n      at: %s" % (
                (size, self.last_free)
                + cuda.mem_get_info()
                + (self.held_blocks, self.active_blocks,
                    description))

        hist_items = sorted(list(histogram.iteritems()))
        for (bsize, descr), count in hist_items:
            print >> self.logfile, \
                    "  %s (%d bytes): %dx" % (descr[len(cpfx):], bsize, count)

        if self.interactive:
            raw_input("  [Enter]")

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
                print size, stack
                while True:
                    mnemonic = raw_input("Enter mnemonic or [Enter] for more info:")
                    if mnemonic == '':
                        from traceback import print_stack
                        print_stack()
                    else:
                        break
                self.stacktrace_mnemonics[stack, size] = mnemonic
                return mnemonic




def _exact_div(dividend, divisor):
    quot, rem = divmod(dividend, divisor)
    assert rem == 0
    return quot

def _int_ceiling(value, multiple_of=1):
    """Round C{value} up to be a C{multiple_of} something."""
    # Mimicks the Excel "floor" function (for code stolen from occupany calculator)

    from math import ceil
    return int(ceil(value/multiple_of))*multiple_of

def _int_floor(value, multiple_of=1):
    """Round C{value} down to be a C{multiple_of} something."""
    # Mimicks the Excel "floor" function (for code stolen from occupany calculator)

    from math import floor
    return int(floor(value/multiple_of))*multiple_of




def get_default_device(default=0):
    from warnings import warn
    warn("get_default_device() is deprecated; "
            "use make_default_context() instead", DeprecationWarning)

    from pycuda.driver import Device
    import os
    dev = os.environ.get("CUDA_DEVICE")

    if dev is None:
        try:
            dev = (open(os.path.join(os.path.expanduser("~"), ".cuda_device"))
                    .read().strip())
        except:
            pass

    if dev is None:
        dev = default

    try:
        dev = int(dev)
    except TypeError:
        raise TypeError("CUDA device number (CUDA_DEVICE or ~/.cuda-device) must be an integer")

    return Device(dev)




def make_default_context():
    ndevices = cuda.Device.count()
    if ndevices == 0:
        errmsg = "No CUDA enabled device found. Please check your installation."
        raise RuntimeError, errmsg

    ndevices = cuda.Device.count()
    if ndevices == 0:
        raise RuntimeError("No CUDA enabled device found. "
                "Please check your installation.")

    # Is CUDA_DEVICE set?
    import os
    devn = os.environ.get("CUDA_DEVICE")

    # Is $HOME/.cuda_device set ?
    if devn is None:
        try:
            homedir = os.environ.get("HOME")
            assert homedir is not None
            devn = (open(os.path.join(homedir, ".cuda_device"))
                    .read().strip())
        except:
            pass

    # If either CUDA_DEVICE or $HOME/.cuda_device is set, try to use it ;-)
    if devn is not None:
        try:
            devn = int(devn)
        except TypeError:
            raise TypeError("CUDA device number (CUDA_DEVICE or ~/.cuda_device)"
                    " must be an integer")

        dev = cuda.Device(devn)
        return dev.make_context()

    # Otherwise, try to use any available device
    else:
        selected_device = None
        for devn in xrange(ndevices):
            dev = cuda.Device(devn)
            try:
                return dev.make_context()
            except cuda.Error:
                pass

        raise RuntimeError("make_default_context() wasn't able to create a context "
                "on any of the %d detected devices" % ndevices)




class DeviceData:
    def __init__(self, dev=None):
        import pycuda.driver as drv

        if dev is None:
            dev = cuda.Context.get_device()

        self.max_threads = dev.get_attribute(drv.device_attribute.MAX_THREADS_PER_BLOCK)
        self.warp_size = dev.get_attribute(drv.device_attribute.WARP_SIZE)

        if dev.compute_capability() < (1,2):
            self.warps_per_mp = 24
        else:
            self.warps_per_mp = 32

        self.thread_blocks_per_mp = 8
        self.registers = dev.get_attribute(drv.device_attribute.MAX_REGISTERS_PER_BLOCK)
        self.shared_memory = dev.get_attribute(drv.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)

        self.smem_granularity = 16

    def align(self, bytes, word_size=4):
        return _int_ceiling(bytes, self.align_bytes(word_size))

    def align_dtype(self, elements, dtype_size):
        return _int_ceiling(elements,
                self.align_words(dtype_size))

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
            raise ValueError, "no alignment possible for fetches of size %d" % word_size

    def coalesce(self, thread_count):
        return _int_ceiling(thread_count, 16)

    @staticmethod
    def make_valid_tex_channel_count(size):
        valid_sizes = [1,2,4]
        for vs in valid_sizes:
            if size <= vs:
                return vs

        raise ValueError, "could not enlarge argument to valid channel count"




class OccupancyRecord:
    def __init__(self, devdata, threads, shared_mem=0, registers=0):
        if threads > devdata.max_threads:
            raise ValueError("too many threads")

        # copied literally from occupancy calculator
        alloc_warps = _int_ceiling(threads/devdata.warp_size)
        alloc_regs = _int_ceiling(alloc_warps*2, 4)*16*registers
        alloc_smem = _int_ceiling(shared_mem, 512)

        if alloc_regs > devdata.registers:
            raise ValueError("too many registers")

        if alloc_smem > devdata.shared_memory:
            raise ValueError("too much smem")

        self.tb_per_mp_limits = [(devdata.thread_blocks_per_mp, "device"),
                (_int_floor(devdata.warps_per_mp/alloc_warps), "warps")
                ]
        if registers > 0:
            self.tb_per_mp_limits.append((_int_floor(devdata.registers/alloc_regs), "regs"))
        if shared_mem > 0:
            self.tb_per_mp_limits.append((_int_floor(devdata.shared_memory/alloc_smem), "smem"))

        self.tb_per_mp, self.limited_by = min(self.tb_per_mp_limits)

        self.warps_per_mp = self.tb_per_mp * alloc_warps
        self.occupancy = self.warps_per_mp / devdata.warps_per_mp




def allow_user_edit(s, filename, descr="the file"):
    from tempfile import mkdtemp
    tempdir = mkdtemp()

    from os.path import join
    full_name = join(tempdir, filename)

    outf = open(full_name, "w")
    outf.write(str(s))
    outf.close()

    raw_input("Edit %s at %s now, then hit [Enter]:"
            % (descr, full_name))

    inf = open(full_name, "r")
    result = inf.read()
    inf.close()

    return result




# C code generation helpers ---------------------------------------------------
@memoize
def platform_bits():
    return tuple.__itemsize__ * 8




def dtype_to_ctype(dtype, with_fp_tex_hack=False):
    if dtype is None:
        raise ValueError("dtype may not be None")

    import numpy
    dtype = numpy.dtype(dtype)
    if dtype == numpy.int64 and platform_bits() == 64:
        return "long"
    elif dtype == numpy.uint64 and platform_bits() == 64:
        return "unsigned long"
    elif dtype == numpy.int32:
        return "int"
    elif dtype == numpy.uint32:
        return "unsigned int"
    elif dtype == numpy.int16:
        return "short int"
    elif dtype == numpy.uint16:
        return "short unsigned int"
    elif dtype == numpy.int8:
        return "signed char"
    elif dtype == numpy.uint8:
        return "unsigned char"
    elif dtype == numpy.bool:
        return "bool"
    elif dtype == numpy.float32:
        if with_fp_tex_hack:
            return "fp_tex_float"
        else:
            return "float"
    elif dtype == numpy.float64:
        if with_fp_tex_hack:
            return "fp_tex_double"
        else:
            return "double"
    elif dtype == numpy.complex64:
        return "pycuda::complex<float>"
    elif dtype == numpy.complex128:
        return "pycuda::complex<double>"
    else:
        raise ValueError, "unable to map dtype '%s'" % dtype




# C argument lists ------------------------------------------------------------
class Argument:
    def __init__(self, dtype, name):
        self.dtype = numpy.dtype(dtype)
        self.name = name

    def __repr__(self):
        return "%s(%r, %s)" % (
                self.__class__.__name__,
                self.name,
                self.dtype)

class VectorArg(Argument):
    def declarator(self):
        return "%s *%s" % (dtype_to_ctype(self.dtype), self.name)

    struct_char = "P"

class ScalarArg(Argument):
    def declarator(self):
        return "%s %s" % (dtype_to_ctype(self.dtype), self.name)

    @property
    def struct_char(self):
        return self.dtype.char





def parse_c_arg(c_arg):
    c_arg = c_arg.replace("const", "").replace("volatile", "")

    # process and remove declarator
    import re
    decl_re = re.compile(r"(\**)\s*([_a-zA-Z0-9]+)(\s*\[[ 0-9]*\])*\s*$")
    decl_match = decl_re.search(c_arg)

    if decl_match is None:
        raise ValueError("couldn't parse C declarator '%s'" % c_arg)

    name = decl_match.group(2)

    if decl_match.group(1) or decl_match.group(3) is not None:
        arg_class = VectorArg
    else:
        arg_class = ScalarArg

    tp = c_arg[:decl_match.start()]
    tp = " ".join(tp.split())

    if tp == "float": dtype = numpy.float32
    elif tp == "double": dtype = numpy.float64
    elif tp == "pycuda::complex<float>": dtype = numpy.complex64
    elif tp == "pycuda::complex<double>": dtype = numpy.complex128
    elif tp in ["int", "signed int"]: dtype = numpy.int32
    elif tp in ["unsigned", "unsigned int"]: dtype = numpy.uint32
    elif tp in ["long", "long int"]:
        if platform_bits() == 64:
            dtype = numpy.int64
        else:
            dtype = numpy.int32
    elif tp in ["unsigned long", "unsigned long int"]:
        if platform_bits() == 64:
            dtype = numpy.uint64
        else:
            dtype = numpy.uint32
    elif tp in ["short", "short int"]: dtype = numpy.int16
    elif tp in ["unsigned short", "unsigned short int"]: dtype = numpy.uint16
    elif tp in ["char"]: dtype = numpy.int8
    elif tp in ["unsigned char"]: dtype = numpy.uint8
    elif tp in ["bool"]: dtype = numpy.bool
    else: raise ValueError, "unknown type '%s'" % tp

    return arg_class(dtype, name)




def get_arg_type(c_arg):
    return parse_c_arg(c_arg).struct_char




@decorator
def context_dependent_memoize(func, *args):
    try:
        ctx_dict = func._pycuda_ctx_dep_memoize_dic
    except AttributeError:
        # FIXME: This may keep contexts alive longer than desired.
        # But I guess since the memory in them is freed, who cares.
        ctx_dict = func._pycuda_ctx_dep_memoize_dic = {}

    cur_ctx = cuda.Context.get_current()

    try:
        return ctx_dict[cur_ctx][args]
    except KeyError:
        arg_dict = ctx_dict.setdefault(cur_ctx, {})
        result = func(*args)
        arg_dict[args] = result
        return result




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
            ctx.detach()

    try:
        from py.test import mark as mark_test
    except ImportError:
        return f

    return mark_test.cuda(f)
