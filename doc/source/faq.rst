Frequently Asked Questions
==========================

How about multiple GPUs?
------------------------

Two ways:

* Allocate two contexts, juggle (:meth:`pycuda.driver.Context.push` and
  :meth:`pycuda.driver.Context.pop`) them from that one process.
* Work with several threads. As of Version 0.90.2, PyCuda will actually 
  release the `GIL <http://en.wikipedia.org/wiki/Global_Interpreter_Lock>`_
  while it is waiting for CUDA operations to finish.

My program terminates after a launch failure. Why?
--------------------------------------------------

You're probably seeing something like this::

  Traceback (most recent call last):
    File "fail.py", line 32, in <module>
      cuda.memcpy_dtoh(a_doubled, a_gpu)
  RuntimeError: cuMemcpyDtoH failed: launch failed
  terminate called after throwing an instance of 'std::runtime_error'
    what():  cuMemFree failed: launch failed
  zsh: abort      python fail.py

What's going on here? First of all, recall that launch failures in 
CUDA are asynchronous. So the actual traceback does not point to
the failed kernel launch, it points to the next CUDA request after
the failed kernel.

Next, as far as I can tell, a CUDA context becomes invalid after a launch
failure, and all following CUDA calls in that context fail. Now, that includes
cleanup (see the :cfunc:`cuMemFree` in the traceback?) that PyCuda tries to perform
automatically. Here, a bit of PyCuda's C++ heritage shows through. While 
performing cleanup, we are processing an exception (the launch failure
reported by :cfunc:`cuMemcpyDtoH`). If another exception occurs during 
exception processing, C++ gives up and aborts the program with a message.

In principle, this could be handled better. If you're willing to dedicate time
to this, I'll likely take your patch.

Are the CUBLAS APIs available via PyCuda?  
-----------------------------------------

No. I would be more than happy to make them available, but that would be mostly
either-or with the rest of PyCuda, because of the following sentence in the
CUDA programming guide:

   [CUDA] is composed of two APIs:

   * A low-level API called the CUDA driver API,
   * A higher-level API called the CUDA runtime API that is implemented on top of
     the CUDA driver API.

   These APIs are mutually exclusive: An application should use either one or the
   other.

PyCuda is based on the driver API. CUBLAS uses the high-level API. Once *can*
violate this rule without crashing immediately. But sketchy stuff does happen.
Instead, for BLAS-1 operations, PyCuda comes with a class called
:class:`pycuda.gpuarray.GPUArray` that essentially reimplements that part of
CUBLAS.

If you dig into the history of PyCuda, you'll find that, at one point, I
did have rudimentary CUBLAS wrappers. I removed them because of the above
issue. If you would like to make CUBLAS wrappers, feel free to use these
rudiments as a starting point. That said, Arno Pähler's python-cuda has
complete :mod:`ctypes`-based wrappers for CUBLAS. I don't think they interact natively
with numpy, though.

I've found some nice undocumented function in PyCuda. Can I use it?
-------------------------------------------------------------------

Of course you can. But don't come whining if it breaks or goes away in
a future release. Being open-source, neither of these two should be
show-stoppers anyway, and we welcome fixes for any functionality,
documented or not.

The rule is that if something is documented, we will in general make
every effort to keep future version backward compatible with the present
interface. If it isn't, there's no such guarantee.

I have <insert random compilation problem> with gcc 4.1 or older. Help!
-----------------------------------------------------------------------

Try adding::

    CXXFLAGS = ['-DBOOST_PYTHON_NO_PY_SIGNATURES']

to your :file:`pycuda/siteconf.py` or :file:`$HOME/.aksetup-defaults.py`.

User-visible Changes
====================

Version 0.92
------------

.. note::

    If you're upgrading from prior versions, 
    you may delete the directory :file:`$HOME/.pycuda-compiler-cache`
    to recover now-unused disk space.

.. note::

    During this release time frame, I had the honor of giving a talk on PyCuda
    for a `class <http://sites.google.com/site/cudaiap2009/>`_ that a group around 
    Nicolas Pinto was teaching at MIT.
    If you're interested, the slides for it are 
    `available <http://mathema.tician.de/dl/pub/pycuda-mit.pdf>`_.

.. warning:: 

    Version 0.92 is currently a release candidate and therefore has a 
    somewhat higher likelihood of bugs.

* Make :class:`pycuda.tools.DeviceMemoryPool` official functionality,
  after numerous improvements. Add :class:`pycuda.tools.PageLockedMemoryPool`
  for pagelocked memory, too.
* Properly deal with automatic cleanup in the face of several contexts.
* Fix compilation on Python 2.4.
* Fix 3D arrays. (Nicolas Pinto)
* Improve error message when :command:`nvcc` is not found.
* Automatically run Python GC before throwing out-of-memory errors.
* Allow explicit release of memory using 
  :meth:`pycuda.driver.DeviceAllocation.free`,
  :meth:`pycuda.driver.HostAllocation.free`,
  :meth:`pycuda.driver.Array.free`,
  :meth:`pycuda.tools.PooledDeviceAllocation.free`,
  :meth:`pycuda.tools.PooledHostAllocation.free`.
* Make configure switch ``./configure.py --cuda-trace`` to enable API tracing.
* Add a documentation chapter and examples on :ref:`metaprog`.
* Add :func:`pycuda.gpuarray.empty_like` and 
  :func:`pycuda.gpuarray.zeros_like`.
* Add and document :attr:`pycuda.gpuarray.GPUArray.mem_size` in anticipation of
  stride/pitch support in :class:`pycuda.gpuarray.GPUArray`.
* Merge Jozef Vesely's MD5-based RNG.
* Document :func:`pycuda.driver.from_device` 
  and :func:`pycuda.driver.from_device_like`.
* Add :class:`pycuda.elementwise.ElementwiseKernel`.
* Various documentation improvements. (many of them from Nicholas Tung)
* Move PyCuda's compiler cache to the system temporary directory, rather
  than the users home directory.

Version 0.91
------------

* Add support for compiling on CUDA 1.1. 
  Added version query :func:`pycuda.driver.get_version`.
  Updated documentation to show 2.0-only functionality.
* Support for Windows and MacOS X, in addition to Linux. 
  (Gert Wohlgemuth, Cosmin Stejerean, Znah on the Nvidia forums,
  and David Gadling)
* Support more arithmetic operators on :class:`pycuda.gpuarray.GPUArray`. (Gert Wohlgemuth)
* Add :func:`pycuda.gpuarray.arange`. (Gert Wohlgemuth)
* Add :mod:`pycuda.curandom`. (Gert Wohlgemuth)
* Add :mod:`pycuda.cumath`. (Gert Wohlgemuth)
* Add :mod:`pycuda.autoinit`.
* Add :mod:`pycuda.tools`.
* Add :class:`pycuda.tools.DeviceData` and :class:`pycuda.tools.OccupancyRecord`.
* :class:`pycuda.gpuarray.GPUArray` parallelizes properly on 
  GTX200-generation devices.
* Make :class:`pycuda.driver.Function` resource usage available
  to the program. (See, e.g. :attr:`pycuda.driver.Function.registers`.)
* Cache kernels compiled by :class:`pycuda.driver.SourceModule`.
  (Tom Annau)
* Allow for faster, prepared kernel invocation. 
  See :meth:`pycuda.driver.Function.prepare`. 
* Added memory pools, at :class:`pycuda.tools.DeviceMemoryPool` as
  experimental, undocumented functionality.
  For some workloads, this can cure the slowness of 
  :func:`pycuda.driver.mem_alloc`.
* Fix the :ref:`memset <memset>` family of functions.
* Improve :ref:`errors`.
* Add `order` parameter to :func:`pycuda.driver.matrix_to_array` and
  :func:`pycuda.driver.make_multichannel_2d_array`.

Acknowledgments
================

* Gert Wohlgemuth ported PyCuda to MacOS X and contributed large parts of
  :class:`pycuda.gpuarray.GPUArray`.
* Znah on the Nvidia forums contributed fixes for Windows XP.
* Cosmin Stejerean provided multiple patches for PyCuda's build system.
* Tom Annau contributed an alternative SourceModule compiler cache as well
  as Windows build insight.
* Nicholas Tung improved PyCuda's documentation.
* Jozef Vesely contributed a massively improved random number generator derived from 
  the RSA Data Security, Inc. MD5 Message Digest Algorithm.

Licensing
=========

PyCuda is licensed to you under the MIT/X Consortium license:

Copyright (c) 2009 Andreas Klöckner and Contributors.

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
