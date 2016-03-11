Changes
=======

Version 2016.2
--------------
.. note::

    This version is the current development version. You can get it from
    `PyCUDA's version control repository <https://github.com/inducer/pycuda>`_.

Version 2016.1
--------------

* Bug fixes.
* Global control of caching.
* Matrix/array interop.
* Add :meth:`pycuda.gpuarray.GPUArray.squeeze`

Version 2014.1
--------------

* Add :meth:`PointerHolderBase.as_buffer` and :meth:`DeviceAllocation.as_buffer`.
* Support for :class:`device_attribute` values added in CUDA 5.0, 5.5, and 6.0.
* Support for :ref:`managed_memory`. (contributed by Stan Seibert)

Version 2013.1.1
----------------

* Windows fix for PyCUDA on Python 3 (Thanks, Christoph Gohlke)

Version 2013.1
--------------

* Python 3 support (large parts contributed by Tomasz Rybak)
* Add :meth:`pycuda.gpuarray.GPUArray.__getitem__`,
  supporting generic slicing.

  It is *possible* to create non-contiguous arrays using this functionality.
  Most operations (elementwise etc.) will not work on such arrays.
* More generators in :mod:`pycuda.curandom`. (contributed by Tomasz Rybak)
* Many bug fixes

.. note::

    The addition of :meth:`pyopencl.array.Array.__getitem__` has an unintended
    consequence due to `numpy bug 3375
    <https://github.com/numpy/numpy/issues/3375>`_.  For instance, this
    expression::

        numpy.float32(5) * some_gpu_array

    may take a very long time to execute. This is because :mod:`numpy` first
    builds an object array of (compute-device) scalars (!) before it decides that
    that's probably not such a bright idea and finally calls
    :meth:`pycuda.gpuarray.GPUArray.__rmul__`.

    Note that only left arithmetic operations of :class:`pycuda.gpuarray.GPUArray`
    by :mod:`numpy` scalars are affected. Python's number types (:class:`float` etc.)
    are unaffected, as are right multiplications.

    If a program that used to run fast suddenly runs extremely slowly, it is
    likely that this bug is to blame.

    Here's what you can do:

    * Use Python scalars instead of :mod:`numpy` scalars.
    * Switch to right multiplications if possible.
    * Use a patched :mod:`numpy`. See the bug report linked above for a pull
      request with a fix.
    * Switch to a fixed version of :mod:`numpy` when available.

Version 2012.1
--------------

* Numerous bug fixes. (including shipped-boost compilation on gcc 4.7)

Version 2011.2
--------------

* Fix a memory leak when using pagelocked memory. (reported by Paul Cazeaux)
* Fix complex scalar argument passing.
* Fix :func:`pycuda.gpuarray.zeros` when used on complex arrays.
* Add :func:`pycuda.tools.register_dtype` to enable scan/reduction on struct types.
* More improvements to CURAND.
* Add support for CUDA 4.1.

Version 2011.1.2
----------------

* Various fixes.

Version 2011.1.1
----------------

* Various fixes.

Version 2011.1
--------------

When you update code to run on this version of PyCUDA, please make sure
to have deprecation warnings enabled, so that you know when your code needs
updating. (See
`the Python docs <http://docs.python.org/dev/whatsnew/2.7.html#the-future-for-python-2-x>`_.
Caution: As of Python 2.7, deprecation warnings are disabled by default.)

* Add support for CUDA 3.0-style OpenGL interop. (thanks to Tomasz Rybak)
* Add :meth:`pycuda.driver.Stream.wait_for_event`.
* Add *range* and *slice* keyword argument to :meth:`pycuda.elementwise.ElementwiseKernel.__call__`.
* Document *preamble* constructor keyword argument to
  :class:`pycuda.elementwise.ElementwiseKernel`.
* Add vector types, see :class:`pycuda.gpuarray.vec`.
* Add :mod:`pycuda.scan`.
* Add support for new features in CUDA 4.0.
* Add :attr:`pycuda.gpuarray.GPUArray.strides`, :attr:`pycuda.gpuarray.GPUArray.flags`.
  Allow the creation of arrys in C and Fortran order.
* Adopt stateless launch interface from CUDA, deprecate old one.
* Add CURAND wrapper. (with work by Tomasz Rybak)
* Add :data:`pycuda.compiler.DEFAULT_NVCC_FLAGS`.

Version 0.94.2
--------------

* Fix the pesky Fermi reduction bug. (thanks to Tomasz Rybak)

Version 0.94.1
--------------

* Support for CUDA debugging.
  (see `FAQ <http://wiki.tiker.net/PyCuda/FrequentlyAskedQuestions>`_ for details.)

Version 0.94
------------

* Support for CUDA 3.0. (but not CUDA 3.0 beta!)
  Search for "CUDA 3.0" in :ref:`reference-doc` to see what's new.
* Support for CUDA 3.1 beta.
  Search for "CUDA 3.1" in :ref:`reference-doc` to see what's new.
* Support for CUDA 3.2 RC.
  Search for "CUDA 3.2" in :ref:`reference-doc` to see what's new.
* Add sparse matrix-vector multiplication and linear system solving code,
  in :mod:`pycuda.sparse`.
* Add :func:`pycuda.gpuarray.if_positive`, :func:`pycuda.gpuarray.maximum`,
  :func:`pycuda.gpuarray.minimum`.
* Deprecate :func:`pycuda.tools.get_default_device`
* Add :func:`pycuda.tools.make_default_context`.
* Use :func:`pycuda.tools.make_default_context` in :mod:`pycuda.autoinit`,
  which changes its behavior.
* Remove previously deprecated features:

  * :attr:`pycuda.driver.Function.registers`,
    :attr:`pycuda.driver.Function.lmem`, and
    :attr:`pycuda.driver.Function.smem` have been deprecated in favor of the
    mechanism above. See :attr:`pycuda.driver.Function.num_regs` for more.
  * the three-argument forms (i.e. with streams)
    of :func:`pycuda.driver.memcpy_dtoh` and
    :func:`pycuda.driver.memcpy_htod`. Use
    :func:`pycuda.driver.memcpy_dtoh_async`
    and :func:`pycuda.driver.memcpy_htod_async` instead.
  * :class:`pycuda.driver.SourceModule`.

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
* Add :class:`pycuda.driver.PointerHolderBase`.

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
* Min Ragan-Kelley solved the long-standing puzzle of why PyCUDA
  did not work on 64-bit CUDA on OS X (and provided a patch).
* Tomasz Rybak solved another long-standing puzzle of why reduction
  failed to work on some Fermi chips. In addition, he provided
  a patch that updated PyCUDA's :ref:`gl-interop` to the state of
  CUDA 3.0.
* Martin Bergtholdt of Philips Research provided a patch that made PyCUDA work
  on 64-bit Windows 7.

Licensing
=========

PyCUDA is licensed to you under the MIT/X Consortium license:

Copyright (c) 2009,10 Andreas Klöckner and Contributors.

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

PyCUDA includes derivatives of parts of the `Thrust
<https://code.google.com/p/thrust/>`_ computing package (in particular the scan
implementation). These parts are licensed as follows:

    Copyright 2008-2011 NVIDIA Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        <http://www.apache.org/licenses/LICENSE-2.0>

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

.. note::

    If you use Apache-licensed parts, be aware that these may be incompatible
    with software licensed exclusively under GPL2.  (Most software is licensed
    as GPL2 or later, in which case this is not an issue.)


Frequently Asked Questions
==========================

The FAQ is now maintained collaboratively in the
`PyCUDA Wiki <http://wiki.tiker.net/PyCuda/FrequentlyAskedQuestions>`_.

Citing PyCUDA
===============

We are not asking you to gratuitously cite PyCUDA in work that is otherwise
unrelated to software. That said, if you do discuss some of the development
aspects of your code and would like to highlight a few of the ideas behind
PyCUDA, feel free to cite `this article
<http://dx.doi.org/10.1016/j.parco.2011.09.001>`_:

    Andreas Klöckner, Nicolas Pinto, Yunsup Lee, Bryan Catanzaro, Paul Ivanov,
    Ahmed Fasih, PyCUDA and PyOpenCL: A scripting-based approach to GPU
    run-time code generation, Parallel Computing, Volume 38, Issue 3, March
    2012, Pages 157-174.

Here's a Bibtex entry for your convenience::

    @article{kloeckner_pycuda_2012,
       author = {{Kl{\"o}ckner}, Andreas
            and {Pinto}, Nicolas
            and {Lee}, Yunsup
            and {Catanzaro}, B.
            and {Ivanov}, Paul
            and {Fasih}, Ahmed },
       title = "{PyCUDA and PyOpenCL: A Scripting-Based Approach to GPU Run-Time Code Generation}",
       journal = "Parallel Computing",
       volume = "38",
       number = "3",
       pages = "157--174",
       year = "2012",
       issn = "0167-8191",
       doi = "10.1016/j.parco.2011.09.001",
    }
