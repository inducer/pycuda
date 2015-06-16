from __future__ import division
from __future__ import absolute_import
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray




STREAM_POOL = []




def get_stream():
    if STREAM_POOL:
        return STREAM_POOL.pop()
    else:
        return drv.Stream()





class AsyncInnerProduct:
    def __init__(self, a, b, pagelocked_allocator):
        self.gpu_result = gpuarray.dot(a, b)
        self.gpu_finished_evt = drv.Event()
        self.gpu_finished_evt.record()
        self.gpu_finished = False

        self.pagelocked_allocator = pagelocked_allocator

    def get_host_result(self):
        if not self.gpu_finished:
            if self.gpu_finished_evt.query():
                self.gpu_finished = True
                self.copy_stream = get_stream()
                self.host_dest = self.pagelocked_allocator(
                        self.gpu_result.shape, self.gpu_result.dtype,
                        self.copy_stream)
                drv.memcpy_dtoh_async(self.host_dest,
                        self.gpu_result.gpudata,
                        self.copy_stream)
                self.copy_finished_evt = drv.Event()
                self.copy_finished_evt.record()
        else:
            if self.copy_finished_evt.query():
                STREAM_POOL.append(self.copy_stream)
                return self.host_dest




def _at_exit():
    STREAM_POOL[:] = []

import atexit
atexit.register(_at_exit)

