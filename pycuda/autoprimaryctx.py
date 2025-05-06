from __future__ import annotations

import atexit

import pycuda.driver as cuda


# Initialize CUDA
cuda.init()

from pycuda.tools import make_default_context


def _retain_primary_context(dev):
    context = dev.retain_primary_context()
    context.push()
    return context


global context
context = make_default_context(_retain_primary_context)
device = context.get_device()


def _finish_up():
    global context
    context.pop()
    context = None

    from pycuda.tools import clear_context_caches

    clear_context_caches()


atexit.register(_finish_up)
