Built-in Utilities
==================

Automatic Initialization
------------------------

.. module:: pycuda.autoinit

The module :mod:`pycuda.autoinit`,  when imported, automatically performs 
all the steps necessary to get CUDA ready for submission of compute kernels.
It uses :func:`pycuda.tools.make_default_context` to create a compute context.

.. data:: device

  An instance of :class:`pycuda.driver.Device` that was used for automatic
  initialization. 

.. data:: context

  A default-constructed instance of :class:`pycuda.driver.Context` 
  on :data:`device`. This context is created by calling
  :func:`pycuda.tools.make_default_context`.

Choice of Device
----------------

.. module:: pycuda.tools

.. function:: make_default_context()

  Return a :class:`pycuda.driver.Context` instance chosen according to the
  following rules:

   * If the environment variable :envvar:`CUDA_DEVICE` is set, its integer
     value is used as the device number.

   * If the file :file:`.cuda-device` is present in the user's home directory,
     the integer value of its contents is used as the device number.

   * Otherwise, all available CUDA devices are tried in a round-robin fashion.

  An error is raised if this does not lead to a usable context.

  .. versionadded: 0.94

.. function:: get_default_device(default=0)

  Deprecated. Use :func:`make_default_context`.

  Return a :class:`pycuda.driver.Device` instance chosen according to the
  following rules:

   * If the environment variable :envvar:`CUDA_DEVICE` is set, its integer
     value is used as the device number.

   * If the file :file:`.cuda-device` is present in the user's home directory,
     the integer value of its contents is used as the device number.

   * Otherwise, `default` is used as the device number.

  .. versionchanged: 0.94

    Deprecated.

Kernel Caching
--------------

.. function:: context_dependent_memoize(func)

    This decorator caches the result of the decorated function, *if* a 
    subsequent occurs in the same :class:`pycuda.driver.Context`.
    This is useful for caching of kernels.

.. function:: clear_context_caches()

    Empties all context-dependent memoization caches. Also releases
    all held reference contexts. If it is important to you that the
    program detaches from its context, you might need to call this
    function to free all remaining references to your context.

Testing
-------

.. function:: mark_cuda_test(func)

    This function, meant for use with :mod:`py.test`, will mark *func* with a
    "cuda" tag and make sure it has a CUDA context available when invoked.


Device Metadata and Occupancy
-----------------------------

.. class:: DeviceData(dev=None)
  
  Gives access to more information on a device than is available through
  :meth:`pycuda.driver.Device.get_attribute`. If `dev` is `None`, it defaults
  to the device returned by :meth:`pycuda.driver.Context.get_device`.

  .. attribute:: max_threads
  .. attribute:: warp_size
  .. attribute:: warps_per_mp
  .. attribute:: thread_blocks_per_mp
  .. attribute:: registers
  .. attribute:: shared_memory
  .. attribute:: smem_granularity

    The number of threads that participate in banked, simultaneous access
    to shared memory.

  .. attribute:: smem_alloc_granularity

    The size of the smallest possible (non-empty) shared memory allocation.

  .. method:: align_bytes(word_size=4)

    The distance between global memory base addresses that 
    allow accesses of word-size `word_size` bytes to get coalesced.

  .. method:: align(bytes, word_size=4)

    Round up `bytes` to the next alignment boundary as given by :meth:`align_bytes`.

  .. method:: align_words(word_size)

    Return `self.align_bytes(word_size)/word_size`, while checking that the division
    did not yield a remainder.

  .. method:: align_dtype(elements, dtype_size)

    Round up `elements` to the next alignment boundary 
    as given by :meth:`align_bytes`, where each element is assumed to be
    `dtype_size` bytes large.

  .. UNDOC coalesce

  .. staticmethod:: make_valid_tex_channel_count(size)

    Round up `size` to a valid texture channel count.

.. class:: OccupancyRecord(devdata, threads, shared_mem=0, registers=0)

  Calculate occupancy for a given kernel workload characterized by 

  * thread count of `threads`
  * shared memory use of `shared_mem` bytes
  * register use of `registers` 32-bit registers

  .. attribute:: tb_per_mp

    How many thread blocks execute on each multiprocessor.

  .. attribute:: limited_by

    What :attr:`tb_per_mp` is limited by. One of `"device"`, `"warps"`,
    `"regs"`, `"smem"`.

  .. attribute:: warps_per_mp

    How many warps execute on each multiprocessor.

  .. attribute:: occupancy

    A `float` value between 0 and 1 indicating how much of each multiprocessor's
    scheduling capability is occupied by the kernel.

.. _mempool:

Memory Pools
------------

The functions :func:`pycuda.driver.mem_alloc` and
:func:`pycuda.driver.pagelocked_empty` can consume a fairly large amount of
processing time if they are invoked very frequently. For example, code based on
:class:`pycuda.gpuarray.GPUArray` can easily run into this issue because a
fresh memory area is allocated for each intermediate result. Memory pools are a
remedy for this problem based on the observation that often many of the block
allocations are of the same sizes as previously used ones.

Then, instead of fully returning the memory to the system and incurring the 
associated reallocation overhead, the pool holds on to the memory and uses it
to satisfy future allocations of similarly-sized blocks. The pool reacts
appropriately to out-of-memory conditions as long as all memory allocations
are made through it. Allocations performed from outside of the pool may run
into spurious out-of-memory conditions due to the pool owning much or all of
the available memory.

Device-based Memory Pool
^^^^^^^^^^^^^^^^^^^^^^^^

.. class:: PooledDeviceAllocation

    An object representing a :class:`DeviceMemoryPool`-based allocation of
    linear device memory.  Once this object is deleted, its associated device
    memory is freed. 
    :class:`PooledDeviceAllocation` instances can be cast to :class:`int` 
    (and :class:`long`), yielding the starting address of the device memory
    allocated.

    .. method:: free

        Explicitly return the memory held by *self* to the associated memory pool.

    .. method:: __len__

        Return the size of the allocated memory in bytes.

.. class:: DeviceMemoryPool

    A memory pool for linear device memory as allocated using 
    :func:`pycuda.driver.mem_alloc`. (see :ref:`mempool`)

    .. attribute:: held_blocks

        The number of unused blocks being held by this pool.

    .. attribute:: active_blocks

        The number of blocks in active use that have been allocated
        through this pool.

    .. method:: allocate(size)

        Return a :class:`PooledDeviceAllocation` of *size* bytes.

    .. method:: free_held

        Free all unused memory that the pool is currently holding.

    .. method:: stop_holding

        Instruct the memory to start immediately freeing memory returned
        to it, instead of holding it for future allocations.
        Implicitly calls :meth:`free_held`.
        This is useful as a cleanup action when a memory pool falls out
        of use.

Memory Pool for pagelocked memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. class:: PooledHostAllocation

    An object representing a :class:`PageLockedMemoryPool`-based allocation of
    linear device memory.  Once this object is deleted, its associated device
    memory is freed. 

    .. method:: free

        Explicitly return the memory held by *self* to the associated memory pool.

    .. method:: __len__

        Return the size of the allocated memory in bytes.

.. class:: PageLockedAllocator(flags=0)

    Specifies the set of :class:`pycuda.driver.host_alloc_flags` used in its 
    associated :class:`PageLockedMemoryPool`.

.. class:: PageLockedMemoryPool(allocator=PageLockedAllocator())

    A memory pool for pagelocked host memory as allocated using 
    :func:`pycuda.driver.pagelocked_empty`. (see :ref:`mempool`)

    .. attribute:: held_blocks

        The number of unused blocks being held by this pool.

    .. attribute:: active_blocks

        The number of blocks in active use that have been allocated
        through this pool.

    .. method:: allocate(shape, dtype, order="C")

        Return an uninitialized ("empty") :class:`numpy.ndarray` with the given 
        *shape*, *dtype*, and *order*. This array will be backed by a
        :class:`PooledHostAllocation`, which can be found as the ``.base``
        attribute of the array.

    .. method:: free_held

        Free all unused memory that the pool is currently holding.

    .. method:: stop_holding

        Instruct the memory to start immediately freeing memory returned
        to it, instead of holding it for future allocations.
        Implicitly calls :meth:`free_held`.
        This is useful as a cleanup action when a memory pool falls out
        of use.
