Frequently Asked Questions
==========================

How about multiple GPUs?
------------------------

Two ways:

* Allocate two contexts, juggle (:meth:`pycuda.driver.Context.push` and
  :meth:`pycuda.driver.Context.pop`) them from that one process.
* Work with several processes or threads, using MPI, :mod:`multiprocesing` 
  or :mod:`threading`. As of Version 0.90.2, PyCUDA will 
  release the `GIL <http://en.wikipedia.org/wiki/Global_Interpreter_Lock>`_
  while it is waiting for CUDA operations to finish. As of version 0.93,
  PyCUDA will actually *work* when used together with threads. Also see
  :ref:`pycuda_threading`, below.

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
cleanup (see the :cfunc:`cuMemFree` in the traceback?) that PyCUDA tries to perform
automatically. Here, a bit of PyCUDA's C++ heritage shows through. While 
performing cleanup, we are processing an exception (the launch failure
reported by :cfunc:`cuMemcpyDtoH`). If another exception occurs during 
exception processing, C++ gives up and aborts the program with a message.

In principle, this could be handled better. If you're willing to dedicate time
to this, I'll likely take your patch.

Are the CUBLAS APIs available via PyCUDA?  
-----------------------------------------

No. I would be more than happy to make them available, but that would be mostly
either-or with the rest of PyCUDA, because of the following sentence in the
CUDA programming guide:

   [CUDA] is composed of two APIs:

   * A low-level API called the CUDA driver API,
   * A higher-level API called the CUDA runtime API that is implemented on top of
     the CUDA driver API.

   These APIs are mutually exclusive: An application should use either one or the
   other.

PyCUDA is based on the driver API. CUBLAS uses the high-level API. Once *can*
violate this rule without crashing immediately. But sketchy stuff does happen.
Instead, for BLAS-1 operations, PyCUDA comes with a class called
:class:`pycuda.gpuarray.GPUArray` that essentially reimplements that part of
CUBLAS.

If you dig into the history of PyCUDA, you'll find that, at one point, I
did have rudimentary CUBLAS wrappers. I removed them because of the above
issue. If you would like to make CUBLAS wrappers, feel free to use these
rudiments as a starting point. That said, Arno Pähler's python-cuda has
complete :mod:`ctypes`-based wrappers for CUBLAS. I don't think they interact natively
with numpy, though.

I've found some nice undocumented function in PyCUDA. Can I use it?
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

Does PyCUDA automatically activate the right context for the object I'm talking to?
-----------------------------------------------------------------------------------

No. It *does* know which context each object belongs, and it does implicitly
activate contexts for cleanup purposes. Since I'm not entirely sure how costly
context activation is supposed to be, PyCUDA will not juggle contexts for you
if you're talking to an object from a context that's not currently active.
Here's a rule of thumb: As long as you have control over invocation order, you
have to manage contexts yourself. Since you mostly don't have control over
cleanup, PyCUDA manages contexts for you in this case. To make this transparent
to you, the user, PyCUDA will automatically restore the previous context once
it's done cleaning up.

.. _pycuda_threading :

How does PyCUDA handle threading?
---------------------------------

As of version 0.93, PyCUDA supports :mod:`threading`. There is an example of how this
can be done in :file:`examples/multiple_threads.py` in the PyCUDA distribution.
When you use threading in PyCUDA, you should be aware of one peculiarity, though.
Contexts in CUDA are a per-thread affair, and as such all contexts associated with
a thread as well as GPU memory, arrays and other resources in that context will
be automatically freed when the thread exits. PyCUDA will notice this and will not
try to free the corresponding resource--it's already gone after all.

There is another, less intended consequence, though: If Python's garbage collector
finds a PyCUDA object it wishes to dispose of, and PyCUDA, upon trying to free it,
determines that the object was allocated outside of the current thread of execution,
then that object is quietly leaked. This properly handles the above situation, but
it mishandles a situation where:

 * You use reference cycles in a GPU driver thread, necessitating the GC (over just
   regular reference counts).
 * You require cleanup to be performed before thread exit.
 * You rely on PyCUDA to perform this cleanup.

To entirely avoid the problem, do one of the following:

 * Use :mod:`multiprocessing` instead of :mod:`threading`.
 * Explicitly call :meth:`free` on the objects you want cleaned up.

User-visible Changes
====================

Version 0.93
------------

.. note:: 

    Version 0.93 is currently in release candidate status. If you'd 
    like to try a snapshot, you may access PyCUDA's source control 
    archive via the PyCUDA homepage.

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
