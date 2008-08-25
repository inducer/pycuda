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

from pycuda._driver import DeviceMemoryPool 
from pycuda._driver import bitlog2




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




class DeviceData:
    def __init__(self, dev):
        import pycuda.driver as drv

        self.max_threads = dev.get_attribute(drv.device_attribute.MAX_THREADS_PER_BLOCK)
        self.warp_size = dev.get_attribute(drv.device_attribute.WARP_SIZE)

        if dev.compute_capability() < (1,2):
            self.warps_per_mp = 24
        else:
            self.warps_per_mp = 32

        self.thread_blocks_per_mp = 8
        self.registers = dev.get_attribute(drv.device_attribute.MAX_REGISTERS_PER_BLOCK)
        self.shared_memory = dev.get_attribute(drv.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)

    def align(self, bytes):
        return _int_ceiling(bytes, self.align_bytes())

    def align_dtype(self, elements, dtype_size):
        return _int_ceiling(elements, 
                self.align_words(dtype_size))

    def align_words(self, word_size):
        return _exact_div(self.align_bytes(word_size), word_size)

    def align_bytes(self, wordsize=4):
        if wordsize == 4:
            return 64
        elif wordsize == 8:
            return 128
        elif wordsize == 16:
            return 128
        else:
            raise ValueError, "no alignment possible for fetches of size %d" % wordsize

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




def _test_occupancy():
    for threads in range(32, 512, 16):
        for smem in range(1024, 16384+1, 1024):
            occ = Occupancy(threads, smem)
            print "t%d s%d: %f %s" % (threads, smem, occ.occupancy, occ.limited_by)




if __name__ == "__main__":
    import pycuda.driver as drv
    drv.init()

    _test_occupancy()
