User-visible Changes
====================

Version 0.94
------------
.. note::

    Version 0.94 is the current development version. You can get it from
    PyCUDA's version control repository.

* Support for CUDA 3.0. (but not CUDA 3.0 beta!)
  Search for "CUDA 3.0" in :ref:`reference-doc` to see what's new.
* Add sparse matrix-vector multiplication and linear system solving code,
  in :mod:`pycuda.sparse`.
* Add :func:`pycuda.gpuarray.if_positive`, :func:`pycuda.gpuarray.maximum`,
  :func:`pycuda.gpuarray.minimum`.
* Deprecate :func:`pycuda.tools.get_default_device` 
* Add :func:`pycuda.tools.make_default_context`.
* Use :func:`pycuda.tools.make_default_context` in :mod:`pycuda.autoinit`,
  which changes its behavior.
* Remove previously deprecated features:
 + :attr:`pycuda.driver.Function.registers`, 
   :attr:`pycuda.driver.Function.lmem`, and
   :attr:`pycuda.driver.Function.smem` have been deprecated in favor of the
   mechanism above. See :attr:`pycuda.driver.Function.num_regs` for more.
 + the three-argument forms (i.e. with streams)
   of :func:`pycuda.driver.memcpy_dtoh` and
   :func:`pycuda.driver.memcpy_htod`. Use 
   :func:`pycuda.driver.memcpy_dtoh_async`
   and :func:`pycuda.driver.memcpy_htod_async` instead.
 + :class:`pycuda.driver.SourceModule`.

* Add :func:`pycuda.tools.context_dependent_memoize`, use it for
  context-dependent caching of PyCUDA's canned kernels.
* Add :func:`pycuda.tools.mark_cuda_test`.
* Add attributes of :exc:`pycuda.driver.CompileError`.
  (requested by Dan Lepage)
* Add preliminary support for complex numbers.
  (initial discussion with Daniel Fan)
* Add 
  :attr:`pycuda.gpuarray.GPUArray.real`,
  :attr:`pycuda.gpuarray.GPUArray.imag`,
  :meth:`pycuda.gpuarray.GPUArray.conj`.

Version 0.93
------------

.. warning::

    Version 0.93 makes some changes to the PyCUDA programming interface.
    In all cases where documented features were changed, the old usage
    continues to work, but results in a warning. It is recommended that
    you update your code to remove the warning.

* OpenGL interoperability in :mod:`pycuda.gl`.
* Document :meth:`pycuda.gpuarray.GPUArray.__len__`. Change its definition
  to match :mod:`numpy`.
* Add :meth:`pycuda.gpuarray.GPUArray.bind_to_texref_ext`.
* Let :class:`pycuda.gpuarray.GPUArray` operators deal with generic
  data types, including type promotion.
* Add :func:`pycuda.gpuarray.take`.
* Fix thread handling by making internal context stack thread-local.
* Add :class:`pycuda.reduction.ReductionKernel`.
* Add :func:`pycuda.gpuarray.sum`, :func:`pycuda.gpuarray.dot`, 
  :func:`pycuda.gpuarray.subset_dot`.
* Synchronous and asynchronous memory transfers are now separate
  from each other, the latter having an ``_async`` suffix.
  The now-synchronous forms still take a :class:`pycuda.driver.Stream`
  argument, but this practice is deprecated and prints a warning.
* :class:`pycuda.gpuarray.GPUArray` no longer has an associated 
  :class:`pycuda.driver.Stream`.  Asynchronous GPUArray transfers are 
  now separate from synchronous ones and have an ``_async`` suffix.
* Support for features added in CUDA 2.2.
* :class:`pycuda.driver.SourceModule` has been moved to
  :class:`pycuda.compiler.SourceModule`. It is still available by
  the old name, but will print a warning about the impending
  deprecation.
* :meth:`pycuda.driver.Device.get_attribute` with a 
  :class:`pycuda.driver.device_attribute` `attr` can now be spelled
  `dev.attr`, with no further namespace detours. (Suggested by Ian Cullinan)
  Likewise for :meth:`pycuda.driver.Function.get_attribute`
* :attr:`pycuda.driver.Function.registers`, 
  :attr:`pycuda.driver.Function.lmem`, and
  :attr:`pycuda.driver.Function.smem` have been deprecated in favor of the
  mechanism above. See :attr:`pycuda.driver.Function.num_regs` for more.
* Add PyCUDA version query mechanism, see :data:`pycuda.VERSION`.

Version 0.92
------------

.. note::

    If you're upgrading from prior versions, 
    you may delete the directory :file:`$HOME/.pycuda-compiler-cache`
    to recover now-unused disk space.

.. note::

    During this release time frame, I had the honor of giving a talk on PyCUDA
    for a `class <http://sites.google.com/site/cudaiap2009/>`_ that a group around 
    Nicolas Pinto was teaching at MIT.
    If you're interested, the slides for it are 
    `available <http://mathema.tician.de/dl/pub/pycuda-mit.pdf>`_.

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
* Move PyCUDA's compiler cache to the system temporary directory, rather
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
* Cache kernels compiled by :class:`pycuda.compiler.SourceModule`.
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

* Gert Wohlgemuth ported PyCUDA to MacOS X and contributed large parts of
  :class:`pycuda.gpuarray.GPUArray`.
* Alexander Mordvintsev contributed fixes for Windows XP.
* Cosmin Stejerean provided multiple patches for PyCUDA's build system.
* Tom Annau contributed an alternative SourceModule compiler cache as well
  as Windows build insight.
* Nicholas Tung improved PyCUDA's documentation.
* Jozef Vesely contributed a massively improved random number generator derived from 
  the RSA Data Security, Inc. MD5 Message Digest Algorithm.
* Chris Heuser provided a test cases for multi-threaded PyCUDA.
* The reduction templating is based on code by Mark Harris at Nvidia.
* Andrew Wagner provided a test case and contributed the port of the
  convolution example. The original convolution code is based on an
  example provided by Nvidia.
* Hendrik Riedmann contributed the matrix transpose and list selection
  examples.
* Peter Berrington contributed a working example for CUDA-OpenGL
  interoperability.
* Maarten Breddels provided a patch for 'flat-egg' support.
* Nicolas Pinto refactored :mod:`pycuda.autoinit` for automatic device
  finding.
* Ian Ozsvald and Fabrizio Milo provided patches.

Licensing
=========

PyCUDA is licensed to you under the MIT/X Consortium license:

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

Frequently Asked Questions
==========================

The FAQ is now maintained collaboratively in the 
`PyCUDA Wiki <http://wiki.tiker.net/PyCuda/FrequentlyAskedQuestions>`_.

