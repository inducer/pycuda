.. _reference-doc:

Device Interface
================

.. module:: pycuda
.. moduleauthor:: Andreas Kloeckner <inform@tiker.net>

Version Queries
---------------

.. data:: VERSION

    Gives the numeric version of PyCUDA as a variable-length tuple
    of integers. Enables easy version checks such as
    *VERSION >= (0, 93)*.

    Added in PyCUDA 0.93.

.. data:: VERSION_STATUS

    A text string such as `"rc4"` or `"beta"` qualifying the status
    of the release.

    .. versionadded:: 0.93

.. data:: VERSION_TEXT

    The full release name (such as `"0.93rc4"`) in string form.

    .. versionadded:: 0.93

.. module:: pycuda.driver
    :synopsis: Use CUDA devices from Python

.. _errors:

Error Reporting
---------------

.. exception:: Error

    Base class of all PyCuda errors.

.. exception:: CompileError

    Thrown when :class:`pycuda.compiler.SourceModule` compilation fails.

    .. attribute:: msg

        .. versionadded:: 0.94

    .. attribute:: stdout

        .. versionadded:: 0.94

    .. attribute:: stderr

        .. versionadded:: 0.94

    .. attribute:: command_line

        .. versionadded:: 0.94


.. exception:: MemoryError

    Thrown when :func:`mem_alloc` or related functionality fails.

.. exception:: LogicError

    Thrown when PyCuda was confronted with a situation where it is likely
    that the programmer has made a mistake. :exc:`LogicErrors` do not depend
    on outer circumstances defined by the run-time environment.

    Example: CUDA was used before it was initialized.

.. exception:: LaunchError

    Thrown when kernel invocation has failed. (Note that this will often be
    reported by the next call after the actual kernel invocation.)

.. exception:: RuntimeError

    Thrown when a unforeseen run-time failure is encountered that is not
    likely due to programmer error.

    Example: A file was not found.


Constants
---------

.. class:: ctx_flags

    Flags for :meth:`Device.make_context`. CUDA 2.0 and above only.

    .. attribute:: SCHED_AUTO

        If there are more contexts than processors, yield, otherwise spin
        while waiting for CUDA calls to complete.

    .. attribute:: SCHED_SPIN

        Spin while waiting for CUDA calls to complete.

    .. attribute:: SCHED_YIELD

         Yield to other threads while waiting for CUDA calls to complete.

    .. attribute:: SCHED_MASK

        Mask of valid scheduling flags in this bitfield.

    .. attribute:: SCHED_BLOCKING_SYNC

        Use blocking synchronization. CUDA 2.2 and newer.

    .. attribute:: MAP_HOST

        Support mapped pinned allocations. CUDA 2.2 and newer.

    .. attribute:: LMEM_RESIZE_TO_MAX

        Keep local memory allocation after launch. CUDA 3.2 and newer.
        Rumored to decrease Fermi launch overhead?

        .. versionadded:: 2011.1

    .. attribute:: FLAGS_MASK

        Mask of valid flags in this bitfield.


.. class:: event_flags

    Flags for :class:`Event`. CUDA 2.2 and newer.

    .. attribute:: DEFAULT
    .. attribute:: BLOCKING_SYNC
    .. attribute:: DISABLE_TIMING

        CUDA 3.2 and newer.

        .. versionadded:: 0.94

    .. attribute:: INTERPROCESS

        CUDA 4.1 and newer.

        .. versionadded:: 2011.2

.. class:: device_attribute

    .. attribute:: MAX_THREADS_PER_BLOCK
    .. attribute:: MAX_BLOCK_DIM_X
    .. attribute:: MAX_BLOCK_DIM_Y
    .. attribute:: MAX_BLOCK_DIM_Z
    .. attribute:: MAX_GRID_DIM_X
    .. attribute:: MAX_GRID_DIM_Y
    .. attribute:: MAX_GRID_DIM_Z
    .. attribute:: TOTAL_CONSTANT_MEMORY
    .. attribute:: WARP_SIZE
    .. attribute:: MAX_PITCH
    .. attribute:: CLOCK_RATE
    .. attribute:: TEXTURE_ALIGNMENT
    .. attribute:: GPU_OVERLAP
    .. attribute:: MULTIPROCESSOR_COUNT

        CUDA 2.0 and above only.

    .. attribute:: SHARED_MEMORY_PER_BLOCK

        Deprecated as of CUDA 2.0. See below for replacement.

    .. attribute:: MAX_SHARED_MEMORY_PER_BLOCK

        CUDA 2.0 and above only.

    .. attribute:: REGISTERS_PER_BLOCK

        Deprecated as of CUDA 2.0. See below for replacement.

    .. attribute:: MAX_REGISTERS_PER_BLOCK

        CUDA 2.0 and above.

    .. attribute:: KERNEL_EXEC_TIMEOUT

        CUDA 2.2 and above.

    .. attribute:: INTEGRATED

        CUDA 2.2 and above.

    .. attribute:: CAN_MAP_HOST_MEMORY

        CUDA 2.2 and above.

    .. attribute:: COMPUTE_MODE

        CUDA 2.2 and above. See :class:`compute_mode`.

    .. attribute:: MAXIMUM_TEXTURE1D_WIDTH
        MAXIMUM_TEXTURE2D_WIDTH
        MAXIMUM_TEXTURE2D_HEIGHT
        MAXIMUM_TEXTURE3D_WIDTH
        MAXIMUM_TEXTURE3D_HEIGHT
        MAXIMUM_TEXTURE3D_DEPTH
        MAXIMUM_TEXTURE2D_ARRAY_WIDTH
        MAXIMUM_TEXTURE2D_ARRAY_HEIGHT
        MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES

        CUDA 3.0 and above.

        .. versionadded:: 0.94

    .. attribute:: MAXIMUM_TEXTURE2D_LAYERED_WIDTH
        MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
        MAXIMUM_TEXTURE2D_LAYERED_LAYERS
        MAXIMUM_TEXTURE1D_LAYERED_WIDTH
        MAXIMUM_TEXTURE1D_LAYERED_LAYERS

        CUDA 4.0 and above.

        .. versionadded:: 2011.1

    .. attribute:: SURFACE_ALIGNMENT

        CUDA 3.0 (post-beta) and above.

        .. versionadded:: 0.94

    .. attribute:: CONCURRENT_KERNELS

        CUDA 3.0 (post-beta) and above.

        .. versionadded:: 0.94

    .. attribute:: ECC_ENABLED

        CUDA 3.0 (post-beta) and above.

        .. versionadded:: 0.94

    .. attribute:: PCI_BUS_ID

        CUDA 3.2 and above.

        .. versionadded:: 0.94

    .. attribute:: PCI_DEVICE_ID

        CUDA 3.2 and above.

        .. versionadded:: 0.94

    .. attribute:: TCC_DRIVER

        CUDA 3.2 and above.

        .. versionadded:: 0.94

    .. attribute:: MEMORY_CLOCK_RATE
        GLOBAL_MEMORY_BUS_WIDTH
        L2_CACHE_SIZE
        MAX_THREADS_PER_MULTIPROCESSOR
        ASYNC_ENGINE_COUNT
        UNIFIED_ADDRESSING

        CUDA 4.0 and above.

        .. versionadded:: 2011.1

    .. attribute :: MAXIMUM_TEXTURE2D_GATHER_WIDTH
        MAXIMUM_TEXTURE2D_GATHER_HEIGHT
        MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE
        MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE
        MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE
        PCI_DOMAIN_ID
        TEXTURE_PITCH_ALIGNMENT
        MAXIMUM_TEXTURECUBEMAP_WIDTH
        MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH
        MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS
        MAXIMUM_SURFACE1D_WIDTH
        MAXIMUM_SURFACE2D_WIDTH
        MAXIMUM_SURFACE2D_HEIGHT
        MAXIMUM_SURFACE3D_WIDTH
        MAXIMUM_SURFACE3D_HEIGHT
        MAXIMUM_SURFACE3D_DEPTH
        MAXIMUM_SURFACE1D_LAYERED_WIDTH
        MAXIMUM_SURFACE1D_LAYERED_LAYERS
        MAXIMUM_SURFACE2D_LAYERED_WIDTH
        MAXIMUM_SURFACE2D_LAYERED_HEIGHT
        MAXIMUM_SURFACE2D_LAYERED_LAYERS
        MAXIMUM_SURFACECUBEMAP_WIDTH
        MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH
        MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS
        MAXIMUM_TEXTURE1D_LINEAR_WIDTH
        MAXIMUM_TEXTURE2D_LINEAR_WIDTH
        MAXIMUM_TEXTURE2D_LINEAR_HEIGHT
        MAXIMUM_TEXTURE2D_LINEAR_PITCH

        CUDA 4.1 and above.

        .. versionadded:: 2011.2

    .. attribute :: MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH
        MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT
        COMPUTE_CAPABILITY_MAJOR
        COMPUTE_CAPABILITY_MINOR
        MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH

        CUDA 5.0 and above.

        .. versionadded:: 2014.1

    .. attribute :: STREAM_PRIORITIES_SUPPORTED

        CUDA 5.5 and above.

        .. versionadded:: 2014.1

    .. attribute :: GLOBAL_L1_CACHE_SUPPORTED
        LOCAL_L1_CACHE_SUPPORTED
        MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
        MAX_REGISTERS_PER_MULTIPROCESSOR
        MANAGED_MEMORY
        MULTI_GPU_BOARD
        MULTI_GPU_BOARD_GROUP_ID

        CUDA 6.0 and above.

        .. versionadded:: 2014.1

.. class:: pointer_attribute

    .. attribute:: CONTEXT
        MEMORY_TYPE
        DEVICE_POINTER
        HOST_POINTER

    CUDA 4.0 and above.

    .. versionadded:: 2011.1

.. class:: profiler_output_mode

    .. attribute:: KEY_VALUE_PAIR
        CSV

    CUDA 4.0 and above.

    .. versionadded:: 2011.1

.. class:: function_attribute

    Flags for :meth:`Function.get_attribute`. CUDA 2.2 and newer.

    .. attribute:: MAX_THREADS_PER_BLOCK
    .. attribute:: SHARED_SIZE_BYTES
    .. attribute:: CONST_SIZE_BYTES
    .. attribute:: LOCAL_SIZE_BYTES
    .. attribute:: NUM_REGS
    .. attribute:: PTX_VERSION

        CUDA 3.0 (post-beta) and above.

        .. versionadded:: 0.94

    .. attribute:: BINARY_VERSION

        CUDA 3.0 (post-beta) and above.

        .. versionadded:: 0.94

    .. attribute:: MAX

.. class:: func_cache

    See :meth:`Function.set_cache_config`. CUDA 3.0 (post-beta) and above.

    .. versionadded:: 0.94

    .. attribute:: PREFER_NONE
    .. attribute:: PREFER_SHARED
    .. attribute:: PREFER_L1
    .. attribute:: PREFER_EQUAL

        CUDA 4.1 and above.

        .. versionadded:: 2011.2

.. class:: shared_config

    See :meth:`Function.set_shared_config`. CUDA 4.2 and above.

    .. attribute:: DEFAULT_BANK_SIZE
    .. attribute:: FOUR_BYTE_BANK_SIZE
    .. attribute:: EIGHT_BYTE_BANK_SIZE

.. class:: array_format

    .. attribute:: UNSIGNED_INT8
    .. attribute:: UNSIGNED_INT16
    .. attribute:: UNSIGNED_INT32
    .. attribute:: SIGNED_INT8
    .. attribute:: SIGNED_INT16
    .. attribute:: SIGNED_INT32
    .. attribute:: HALF
    .. attribute:: FLOAT

.. class:: array3d_flags

    .. attribute :: 2DARRAY

        CUDA 3.0 and above. Deprecated--use :attr:`LAYERED`.

        .. versionadded:: 0.94

    .. attribute :: LAYERED

        CUDA 4.0 and above.

        .. versionadded:: 2011.1

    .. attribute :: SURFACE_LDST

        CUDA 3.1 and above.

        .. versionadded:: 0.94

    .. attribute :: CUBEMAP TEXTURE_GATHER

        CUDA 4.1 and above.

        .. versionadded:: 2011.2

.. class:: address_mode

    .. attribute:: WRAP
    .. attribute:: CLAMP
    .. attribute:: MIRROR
    .. attribute:: BORDER

        CUDA 3.2 and above.

        .. versionadded:: 0.94

.. class:: filter_mode

    .. attribute:: POINT
    .. attribute:: LINEAR

.. class:: memory_type

    .. attribute:: HOST
    .. attribute:: DEVICE
    .. attribute:: ARRAY

.. class:: compute_mode

    CUDA 2.2 and newer.

    .. attribute:: DEFAULT
    .. attribute:: PROHIBITED
    .. attribute:: EXCLUSIVE_PROCESS

        CUDA 4.0 and above.

        .. versionadded:: 2011.1

.. class:: jit_option

    CUDA 2.1 and newer.

    .. attribute:: MAX_REGISTERS
    .. attribute:: THREADS_PER_BLOCK
    .. attribute:: WALL_TIME
    .. attribute:: INFO_LOG_BUFFER
    .. attribute:: INFO_LOG_BUFFER_SIZE_BYTES
    .. attribute:: ERROR_LOG_BUFFER
    .. attribute:: ERROR_LOG_BUFFER_SIZE_BYTES
    .. attribute:: OPTIMIZATION_LEVEL
    .. attribute:: TARGET_FROM_CUCONTEXT
    .. attribute:: TARGET
    .. attribute:: FALLBACK_STRATEGY

.. class:: jit_target

    CUDA 2.1 and newer.

    .. attribute:: COMPUTE_10
    .. attribute:: COMPUTE_11
    .. attribute:: COMPUTE_12
    .. attribute:: COMPUTE_13
    .. attribute:: COMPUTE_20

        CUDA 3.0 and above.

        .. versionadded:: 0.94

    .. attribute:: COMPUTE_21

        CUDA 3.2 and above.

        .. versionadded:: 0.94

.. class:: jit_fallback

    CUDA 2.1 and newer.

    .. attribute:: PREFER_PTX
    .. attribute:: PREFER_BINARY

.. class:: host_alloc_flags

    Flags to be used to allocate :ref:`pagelocked_memory`.

    .. attribute:: PORTABLE
    .. attribute:: DEVICEMAP
    .. attribute:: WRITECOMBINED

.. class:: mem_attach_flags

    Flags to be used to allocate :ref:`managed_memory`.

    ..versionadded:: 2014.1

    .. attribute:: GLOBAL
    .. attribute:: HOST
    .. attribute:: SINGLE

.. class:: mem_host_register_flags

    .. attribute:: PORTABLE
    .. attribute:: DEVICEMAP

    CUDA 4.0 and newer.

    .. versionadded:: 2011.1

.. class:: limit

    Limit values for :meth:`Context.get_limit` and :meth:`Context.set_limit`.

    CUDA 3.1 and newer.

    .. versionadded:: 0.94

    .. attribute:: STACK_SIZE
    .. attribute:: PRINTF_FIFO_SIZE
    .. attribute:: MALLOC_HEAP_SIE

        CUDA 3.2 and above.

.. class:: ipc_mem_flags

    .. attribute:: LAZY_ENABLE_PEER_ACCESS


Graphics-related constants
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. class:: graphics_register_flags

    .. versionadded:: 2011.1

    CUDA 4.0 and above.

    .. attribute:: NONE READ_ONLY WRITE_DISCARD SURFACE_LDST

    .. attribute :: TEXTURE_GATHER

        CUDA 4.1 and above.

        .. versionadded:: 2011.2


.. class:: array_cubemap_face

    .. attribute::
        POSITIVE_X NEGATIVE_X
        POSITIVE_Y NEGATIVE_Y
        POSITIVE_Z NEGATIVE_Z

    CUDA 3.2 and above.

    .. versionadded:: 2011.1

Devices and Contexts
--------------------

.. function:: get_version()

    Obtain the version of CUDA against which PyCuda was compiled. Returns a
    3-tuple of integers as *(major, minor, revision)*.

.. function:: get_driver_version()

    Obtain the version of the CUDA driver on top of which PyCUDA is
    running. Returns an integer version number.

.. function:: init(flags=0)

    Initialize CUDA.

    .. warning:: This must be called before any other function in this module.

    See also :mod:`pycuda.autoinit`.

.. class:: Device(number)
        Device(pci_bus_id)

    A handle to the *number*'th CUDA device. See also :mod:`pycuda.autoinit`.

    .. versionchanged:: 2011.2
        The *pci_bus_id* version of the constructor is new in CUDA 4.1.

    .. staticmethod:: count()

        Return the number of CUDA devices found.

    .. method:: name()

    .. method:: pci_bus_id()

        CUDA 4.1 and newer.

        .. versionadded:: 2011.2

    .. method:: compute_capability()

        Return a 2-tuple indicating the compute capability version of this device.

    .. method:: total_memory()

        Return the total amount of memory on the device in bytes.

    .. method:: get_attribute(attr)

        Return the (numeric) value of the attribute *attr*, which may be one of the
        :class:`device_attribute` values.

        All :class:`device_attribute` values may also be directly read
        as (lower-case) attributes on the :class:`Device` object itself,
        e.g. `dev.clock_rate`.

    .. method:: get_attributes()

        Return all device attributes in a :class:`dict`, with keys from
        :class:`device_attribute`.

    .. method:: make_context(flags=ctx_flags.SCHED_AUTO)

        Create a :class:`Context` on this device, with flags taken from the
        :class:`ctx_flags` values.

        Also make the newly-created context the current context.

    .. method:: can_access_peer(dev)

        CUDA 4.0 and newer.

        .. versionadded:: 2011.1

    .. method:: __hash__()
    .. method:: __eq__()
    .. method:: __ne__()

.. class:: Context

    An equivalent of a UNIX process on the compute device.
    Create instances of this class using :meth:`Device.make_context`.
    See also :mod:`pycuda.autoinit`.

    .. method:: detach()

        Decrease the reference count on this context. If the reference count
        hits zero, the context is deleted.

    .. method:: push()

        Make *self* the active context, pushing it on top of the context stack.
        CUDA 2.0 and above only.

    .. staticmethod:: pop()

        Remove any context from the top of the context stack, deactivating it.
        CUDA 2.0 and above only.

    .. staticmethod:: get_device()

        Return the device that the current context is working on.

    .. staticmethod:: synchronize()

        Wait for all activity in the current context to cease, then return.

    .. staticmethod:: set_limit(limit, value)

        See :class:`limit` for possible values of *limit*.

        CUDA 3.1 and above.

        .. versionadded:: 0.94

    .. staticmethod:: get_limit(limit)

        See :class:`limit` for possible values of *limit*.

        CUDA 3.1 and above.

        .. versionadded:: 0.94

    .. staticmethod:: set_cache_config(cc)

        See :class:`func_cache` for possible values of *cc*.

        CUDA 3.2 and above.

        .. versionadded:: 0.94

    .. staticmethod:: get_cache_config()

        Return a value from :class:`func_cache`.

        CUDA 3.2 and above.

        .. versionadded:: 0.94

    .. staticmethod:: set_shared_config(sc)

        See :class:`shared_config` for possible values of *sc*.

        CUDA 4.2 and above.

        .. versionadded:: 2013.1

    .. staticmethod:: get_shared_config()

        Return a value from :class:`shared_config`.

        CUDA 4.2 and above.

        .. versionadded:: 2013.1

    .. method:: get_api_version()

        Return an integer API version number.

        CUDA 3.2 and above.

        .. versionadded:: 0.94

    .. method:: enable_peer_access(peer, flags=0)

        CUDA 4.0 and above.

        .. versionadded:: 2011.1

    .. method:: disable_peer_access(peer, flags=0)

        CUDA 4.0 and above.

        .. versionadded:: 2011.1

Concurrency and Streams
-----------------------

.. class:: Stream(flags=0)

    A handle for a queue of operations that will be carried out in order.

    .. method:: synchronize()

        Wait for all activity on this stream to cease, then return.

    .. method:: is_done()

        Return *True* iff all queued operations have completed.

    .. method:: wait_for_event(evt)

        Enqueues a wait for the given :class:`Event` instance.

        CUDA 3.2 and above.

        .. versionadded:: 2011.1

.. class:: Event(flags=0)

    An event is a temporal 'marker' in a :class:`Stream` that allows taking the time
    between two events--such as the time required to execute a kernel.
    An event's time is recorded when the :class:`Stream` has finished all tasks
    enqueued before the :meth:`record` call.

    See :class:`event_flags` for values for the *flags* parameter.

    .. method:: record(stream=None)

        Insert a recording point for *self* into the :class:`Stream` *stream*.
        Return *self*.

    .. method:: synchronize()

        Wait until the device execution stream reaches this event.
        Return *self*.

    .. method:: query()

        Return *True* if the device execution stream has reached this event.

    .. method:: time_since(event)

        Return the time in milliseconds that has passed between *self* and *event*.
        Use this method as `end.time_since(start)`. Note that this method will fail
        with an "invalid value" error if either of the events has not been reached yet.
        Use :meth:`synchronize` to ensure that the event has been reached.

    .. method:: time_till(event)

        Return the time in milliseconds that has passed between *event* and *self*.
        Use this method as `start.time_till(end)`. Note that this method will fail
        with an "invalid value" error if either of the events has not been reached yet.
        Use :meth:`synchronize` to ensure that the event has been reached.

    .. method:: ipc_handle()

        Return a :class:`bytes` object representing an IPC handle to this event.
        Requires Python 2.6 and CUDA 4.1.

        .. versionadded: 2011.2

    .. staticmethod:: from_ipc_handle(handle)

        Requires Python 2.6 and CUDA 4.1.

        .. versionadded: 2011.2

Memory
------

Global Device Memory
^^^^^^^^^^^^^^^^^^^^

.. function:: mem_get_info()

    Return a tuple *(free, total)* indicating the free and total memory
    in the current context, in bytes.

.. function:: mem_alloc(bytes)

    Return a :class:`DeviceAllocation` object representing a linear
    piece of device memory.

.. function:: to_device(buffer)

    Allocate enough device memory for *buffer*, which adheres to the Python
    :class:`buffer` interface. Copy the contents of *buffer* onto the device.
    Return a :class:`DeviceAllocation` object representing the newly-allocated
    memory.

.. function:: from_device(devptr, shape, dtype, order="C")

    Make a new :class:`numpy.ndarray` from the data at *devptr* on the
    GPU, interpreting them using *shape*, *dtype* and *order*.

.. function:: from_device_like(devptr, other_ary)

    Make a new :class:`numpy.ndarray` from the data at *devptr* on the
    GPU, interpreting them as having the same shape, dtype and order
    as *other_ary*.

.. function:: mem_alloc_pitch(width, height, access_size)

    Allocates a linear piece of device memory at least *width* bytes wide and
    *height* rows high that an be accessed using a data type of size
    *access_size* in a coalesced fashion.

    Returns a tuple *(dev_alloc, actual_pitch)* giving a :class:`DeviceAllocation`
    and the actual width of each row in bytes.

.. class:: DeviceAllocation

    An object representing an allocation of linear device memory.
    Once this object is deleted, its associated device memory is
    freed.

    Objects of this type can be cast to :class:`int` to obtain a linear index
    into this :class:`Context`'s memory.

    .. method:: free()

        Release the held device memory now instead of when this object
        becomes unreachable. Any further use of the object is an error
        and will lead to undefined behavior.

    .. method:: as_buffer(size, offset=0)

        Return the pointer encapsulated by *self* as a Python buffer
        object, with the given *size* and, optionally, *offset*.

        .. versionadded:: 2014.1

.. function:: mem_get_ipc_handle(devptr)

    Return an opaque :class:`bytes` object representing an IPC handle to the
    device pointer *devptr*.

    .. versionadded:: 2011.2

    Requires CUDA 4.1 and Python 2.6.

.. class:: IPCMemoryHandle(ipc_handle, flags=ipc_mem_flags.LAZY_ENABLE_PEER_ACCESS)

    .. versionadded:: 2011.2

    Requires CUDA 4.1 and Python 2.6.

    Objects of this type can be used in the same ways as a
    :class:`DeviceAllocation`.

    .. method:: close()

.. class:: PointerHolderBase

    A base class that facilitates casting to pointers within PyCUDA.
    This allows the user to construct custom pointer types that may
    have been allocated by facilities outside of PyCUDA proper, but
    still need to be objects to facilitate RAII. The user needs to
    supply one method to facilitate the pointer cast:

    .. method:: get_pointer()

        Return the pointer encapsulated by *self*.

    .. method:: as_buffer(size, offset=0)

        Return the pointer encapsulated by *self* as a Python buffer
        object, with the given *size* and, optionally, *offset*.

        .. versionadded:: 2014.1

.. _pagelocked_memory :

Pagelocked Host Memory
^^^^^^^^^^^^^^^^^^^^^^

Pagelocked Allocation
~~~~~~~~~~~~~~~~~~~~~

.. function:: pagelocked_empty(shape, dtype, order="C", mem_flags=0)

    Allocate a pagelocked :class:`numpy.ndarray` of *shape*, *dtype* and *order*.

    *mem_flags* may be one of the values in :class:`host_alloc_flags`.
    It may only be non-zero on CUDA 2.2 and newer.

    For the meaning of the other parameters, please refer to the :mod:`numpy`
    documentation.

.. function:: pagelocked_zeros(shape, dtype, order="C", mem_flags=0)

    Like :func:`pagelocked_empty`, but initialized to zero.

.. function:: pagelocked_empty_like(array, mem_flags=0)

.. function:: pagelocked_zeros_like(array, mem_flags=0)

The :class:`numpy.ndarray` instances returned by these functions
have an attribute *base* that references an object of type

.. class:: PagelockedHostAllocation

    Inherits from :class:`HostPointer`.

    An object representing an allocation of pagelocked
    host memory.  Once this object is deleted, its associated
    device memory is freed.

    .. method:: free()

        Release the held memory now instead of when this object
        becomes unreachable. Any further use of the object (or its
        associated :mod:`numpy` array) is an error
        and will lead to undefined behavior.

    .. method:: get_flags()

        Return a bit field of values from :class:`host_alloc_flags`.

        Only available on CUDA 3.2 and newer.

        .. versionadded:: 0.94

.. class:: HostAllocation

    A deprecated name for :class:`PagelockedHostAllocation`.

.. _aligned_host_memory :

Aligned Host Memory
~~~~~~~~~~~~~~~~~~~

.. function:: aligned_empty(shape, dtype, order="C", alignment=4096)

    Allocate an :class:`numpy.ndarray` of *shape*, *dtype* and *order*,
    with data aligned to *alignment* bytes.

    For the meaning of the other parameters, please refer to the :mod:`numpy`
    documentation.

    .. versionadded:: 2011.1

.. function:: aligned_zeros(shape, dtype, order="C", alignment=4096)

    Like :func:`aligned_empty`, but with initialization to zero.

    .. versionadded:: 2011.1

.. function:: aligned_empty_like(array, alignment=4096)

    .. versionadded:: 2011.1

.. function:: aligned_zeros_like(array, alignment=4096)

    .. versionadded:: 2011.1

The :class:`numpy.ndarray` instances returned by these functions
have an attribute *base* that references an object of type

.. class:: AlignedHostAllocation

    Inherits from :class:`HostPointer`.

    An object representing an allocation of aligned
    host memory.

    .. method:: free()

        Release the held memory now instead of when this object
        becomes unreachable. Any further use of the object (or its
        associated :mod:`numpy` array) is an error
        and will lead to undefined behavior.

Post-Allocation Pagelocking
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: register_host_memory(ary, flags=0)

    Returns a :class:`numpy.ndarray` which shares memory with *ary*.
    This memory will be page-locked as long as the return value of
    this function is alive.

    The returned array's *base* attribute contains a
    :class:`RegisteredHostMemory` instance, whose *base* attribute
    in turn contains *ary*.

    CUDA 4.0 and newer.

    *ary*'s data address and size must be page-aligned. One way to achieve this
    is to use the functions in :ref:`aligned_host_memory`.

    .. versionadded:: 2011.1

.. class:: RegisteredHostMemory

    Inherits from :class:`HostPointer`.

    CUDA 4.0 and newer.

    .. versionadded:: 2011.1

    .. method:: unregister()

        Unregister the page-lock on the host memory held by this instance.
        Note that this does not free the memory, it only frees the
        page-lock.

    .. attribute:: base

        Contains the Python object from which this instance was constructed.

.. class:: HostPointer

    Represents a page-locked host pointer.

    .. method:: get_device_pointer()

        Return a device pointer that indicates the address at which
        this memory is mapped into the device's address space.

        Only available on CUDA 2.2 and newer.

.. _managed_memory :

Managed Memory
^^^^^^^^^^^^^^

CUDA 6.0 adds support for a "Unified Memory" model, which creates a managed
virtual memory space that is visible to both CPUs and GPUs.  The OS will
migrate the physical pages associated with managed memory between the CPU and
GPU as needed.  This allows a numpy array on the host to be passed to kernels
without first creating a DeviceAllocation and manually copying the host data
to and from the device.

.. note::

    Managed memory is only available for some combinations of CUDA device,
    operating system, and host compiler target architecture.  Check the CUDA
    C Programming Guide and CUDA release notes for details.

.. warning::

    This interface to managed memory should be considered experimental. It is
    provided as a preview, but for now the same interface stability guarantees
    as for the rest of PyCUDA do not apply.

Managed Memory Allocation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: managed_empty(shape, dtype, order="C", mem_flags=0)

    Allocate a managed :class:`numpy.ndarray` of *shape*, *dtype* and *order*.

    *mem_flags* may be one of the values in :class:`mem_attach_flags`.

    For the meaning of the other parameters, please refer to the :mod:`numpy`
    documentation.

    Only available on CUDA 6.0 and newer.

    .. versionadded:: 2014.1

.. function:: managed_zeros(shape, dtype, order="C", mem_flags=0)

    Like :func:`managed_empty`, but initialized to zero.

    Only available on CUDA 6.0 and newer.

    .. versionadded:: 2014.1

.. function:: managed_empty_like(array, mem_flags=0)

    Only available on CUDA 6.0 and newer.

    .. versionadded:: 2014.1

.. function:: managed_zeros_like(array, mem_flags=0)

    Only available on CUDA 6.0 and newer.

    .. versionadded:: 2014.1

The :class:`numpy.ndarray` instances returned by these functions
have an attribute *base* that references an object of type

.. class:: ManagedAllocation

    An object representing an allocation of managed
    host memory.  Once this object is deleted, its associated
    CUDA managed memory is freed.

    .. method:: free()

        Release the held memory now instead of when this object
        becomes unreachable. Any further use of the object (or its
        associated :mod:`numpy` array) is an error
        and will lead to undefined behavior.

    .. method:: get_device_pointer()

        Return a device pointer that indicates the address at which
        this memory is mapped into the device's address space.  For
        managed memory, this is also the host pointer.

    .. method:: attach(mem_flags, stream=None)

        Alter the visibility of the managed allocation to be one of the values
        in :class:`mem_attach_flags`.  A managed array can be made visible to
        the host CPU and the entire CUDA context with
        *mem_attach_flags.GLOBAL*, or limited to the CPU only with
        *mem_attach_flags.HOST*.  If *mem_attach_flags.SINGLE* is selected,
        then the array will only be visible to CPU and the provided instance
        of :class:`Stream`.


Managed Memory Usage
~~~~~~~~~~~~~~~~~~~~

A managed numpy array is constructed and used on the host in a similar manner
to a pagelocked array::

    from pycuda.autoinit import context
    import pycuda.driver as cuda
    import numpy as np

    a = cuda.managed_empty(shape=10, dtype=np.float32, mem_flags=cuda.mem_attach_flags.GLOBAL)
    a[:] = np.linspace(0, 9, len(a)) # Fill array on host

It can be passed to a GPU kernel, and used again on the host without
an explicit copy::

    from pycuda.compiler import SourceModule
    mod = SourceModule("""
    __global__ void doublify(float *a)
    {
        a[threadIdx.x] *= 2;
    }
    """)
    doublify = mod.get_function("doublify")

    doublify(a, grid=(1,1), block=(len(a),1,1))
    context.synchronize() # Wait for kernel completion before host access

    median = np.median(a) # Computed on host!

.. warning::

    The CUDA Unified Memory model has very specific rules regarding concurrent
    access of managed memory allocations.  Host access to any managed array
    is not allowed while the GPU is executing a kernel, regardless of whether
    the array is in use by the running kernel.  Failure to follow the
    concurrency rules will generate a segmentation fault, *causing the Python
    interpreter to terminate immediately*.

    Users of managed numpy arrays should read the "Unified Memory Programming"
    appendix of the CUDA C Programming Guide for further details on the
    concurrency restrictions.

    If you are encountering interpreter terminations due to concurrency issues,
    the `faulthandler <http://pypi.python.org/pypi/faulthandler>` module may be
    helpful in locating the location in your Python program where the faulty
    access is occurring.

Arrays and Textures
^^^^^^^^^^^^^^^^^^^

.. class:: ArrayDescriptor

    .. attribute:: width
    .. attribute:: height
    .. attribute:: format

        A value of type :class:`array_format`.

    .. attribute:: num_channels

.. class:: ArrayDescriptor3D

    .. attribute:: width
    .. attribute:: height
    .. attribute:: depth
    .. attribute:: format

        A value of type :class:`array_format`. CUDA 2.0 and above only.

    .. attribute:: num_channels

.. class:: Array(descriptor)

    A 2D or 3D memory block that can only be accessed via
    texture references.

    *descriptor* can be of type :class:`ArrayDescriptor` or
    :class:`ArrayDescriptor3D`.

    .. method::  free()

        Release the array and its device memory now instead of when
        this object becomes unreachable. Any further use of the
        object is an error and will lead to undefined behavior.

    .. method::  get_descriptor()

        Return a :class:`ArrayDescriptor` object for this 2D array,
        like the one that was used to create it.

    .. method::  get_descriptor_3d()

        Return a :class:`ArrayDescriptor3D` object for this 3D array,
        like the one that was used to create it.  CUDA 2.0 and above only.

    .. attribute:: handle

       Return an :class:`int` representing the address in device memory where
       this array resides.


.. class:: SurfaceReference()

    .. note::

        Instances of this class can only be constructed through
        :meth:`Module.get_surfref`.

    CUDA 3.1 and above.

    .. versionadded:: 0.94

    .. method:: set_array(array, flags=0)

        Bind *self* to the :class:`Array` *array*.

        As long as *array* remains bound to this texture reference, it will not be
        freed--the texture reference keeps a reference to the array.

    .. method:: get_array()

        Get back the :class:`Array` to which *self* is bound.

        .. note::

            This will be a different object than the one passed to
            :meth:`set_array`, but it will compare equal.

.. class:: TextureReference()

    A handle to a binding of either linear memory or an :class:`Array` to
    a texture unit.

    .. method:: set_array(array)

        Bind *self* to the :class:`Array` *array*.

        As long as *array* remains bound to this texture reference, it will not be
        freed--the texture reference keeps a reference to the array.

    .. method:: set_address(devptr, bytes, allow_offset=False)

        Bind *self* to the a chunk of linear memory starting at the integer address
        *devptr*, encompassing a number of *bytes*. Due to alignment requirements,
        the effective texture bind address may be different from the requested one
        by an offset. This method returns this offset in bytes. If *allow_offset*
        is ``False``, a nonzero value of this offset will cause an exception to be
        raised.

        Unlike for :class:`Array` objects, no life support is provided for linear memory
        bound to texture references.

    .. method:: set_address_2d(devptr, descr, pitch)

        Bind *self* as a 2-dimensional texture to a chunk of global memory
        at *devptr*. The line-to-line offset in bytes is given by *pitch*.
        Width, height and format are given in the :class:`ArrayDescriptor`
        *descr*. :meth:`set_format` need not and should not be called in
        addition to this method.

    .. method:: set_format(fmt, num_components)

        Set the texture to have :class:`array_format` *fmt* and to have
        *num_components* channels.

    .. method:: set_address_mode(dim, am)

        Set the address mode of dimension *dim* to *am*, which must be one of the
        :class:`address_mode` values.

    .. method:: set_flags(flags)

        Set the flags to a combination of the *TRSF_XXX* values.

    .. method:: get_array()

        Get back the :class:`Array` to which *self* is bound.

        .. note::

            This will be a different object than the one passed to
            :meth:`set_array`, but it will compare equal.

    .. method:: get_address_mode(dim)
    .. method:: get_filter_mode()
    .. method:: get_format()

        Return a tuple *(fmt, num_components)*, where *fmt* is
        of type :class:`array_format`, and *num_components* is the
        number of channels in this texture.

        (Version 2.0 and above only.)

    .. method:: get_flags()

.. data:: TRSA_OVERRIDE_FORMAT
.. data:: TRSF_READ_AS_INTEGER
.. data:: TRSF_NORMALIZED_COORDINATES
.. data:: TR_DEFAULT

.. function:: matrix_to_array(matrix, order)

    Turn the two-dimensional :class:`numpy.ndarray` object *matrix* into an
    :class:`Array`.
    The `order` argument can be either `"C"` or `"F"`. If
    it is `"C"`, then `tex2D(x,y)` is going to fetch `matrix[y,x]`,
    and vice versa for for `"F"`.

.. function:: np_to_array(nparray, order, allowSurfaceBind=False)

    Turn a :class:`numpy.ndarray` with 2D or 3D structure, into an
    :class:`Array`.
    The `order` argument can be either `"C"` or `"F"`.
    If `allowSurfaceBind` is passed as *True* the returned :class:`Array`
    can be read and write with :class:`SurfaceReference` in addition of reads by
    :class:`TextureReference`.
    Function automatically detect *dtype* and adjust channels to
    supported :class:`array_format`. Also add direct support
    for `np.float64`, `np.complex64` and `np.complex128` formats.

    .. highlight:: c

    Example of use::

        #include <pycuda-helpers.hpp>

        texture<fp_tex_double, 3, cudaReadModeElementType> my_tex; // complex128: fp_tex_cdouble
                                                                   // complex64 : fp_tex_cfloat
                                                                   // float64   : fp_tex_double
        surface<void, 3, cudaReadModeElementType> my_surf;         // Surfaces in 2D needs 'cudaSurfaceType2DLayered'

        __global__ void f()
        {
          ...
          fp_tex3D(my_tex, i, j, k);
          fp_surf3Dwrite(myvar, my_surf, i, j, k, cudaBoundaryModeClamp); // fp extensions don't need width in bytes
          fp_surf3Dread(&myvar, my_surf, i, j, k, cudaBoundaryModeClamp);
          ...
        }

    .. versionadded:: 2015.1

.. function:: gpuarray_to_array(gpuparray, order, allowSurfaceBind=False)

    Turn a :class:`GPUArray` with 2D or 3D structure, into an
    :class:`Array`. Same structure and use of :func:`np_to_array`

    .. versionadded:: 2015.1

.. function:: make_multichannel_2d_array(matrix, order)

    Turn the three-dimensional :class:`numpy.ndarray` object *matrix* into
    an 2D :class:`Array` with multiple channels.

    Depending on `order`, the `matrix`'s shape is interpreted as

    * `height, width, num_channels` for `order == "C"`,
    * `num_channels, width, height` for `order == "F"`.

    .. note ::

        This function assumes that *matrix* has been created with
        the memory order *order*. If that is not the case, the
        copied data will likely not be what you expect.

.. _memset:

Initializing Device Memory
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: memset_d8(dest, data, count)
.. function:: memset_d16(dest, data, count)
.. function:: memset_d32(dest, data, count)

    .. note::

        *count* is the number of elements, not bytes.

.. function:: memset_d2d8(dest, pitch, data, width, height)
.. function:: memset_d2d16(dest, pitch, data, width, height)
.. function:: memset_d2d32(dest, pitch, data, width, height)

Unstructured Memory Transfers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: memcpy_htod(dest, src)

    Copy from the Python buffer *src* to the device pointer *dest*
    (an :class:`int` or a :class:`DeviceAllocation`). The size of
    the copy is determined by the size of the buffer.

.. function:: memcpy_htod_async(dest, src, stream=None)

    Copy from the Python buffer *src* to the device pointer *dest*
    (an :class:`int` or a :class:`DeviceAllocation`) asynchronously,
    optionally serialized via *stream*. The size of
    the copy is determined by the size of the buffer.

    *src* must be page-locked memory, see, e.g. :func:`pagelocked_empty`.

    New in 0.93.

.. function:: memcpy_dtoh(dest, src)

    Copy from the device pointer *src* (an :class:`int` or a
    :class:`DeviceAllocation`) to the Python buffer *dest*. The size of the copy
    is determined by the size of the buffer.

.. function:: memcpy_dtoh_async(dest, src, stream=None)

    Copy from the device pointer *src* (an :class:`int` or a
    :class:`DeviceAllocation`) to the Python buffer *dest* asynchronously,
    optionally serialized via *stream*. The size of the copy
    is determined by the size of the buffer.

    *dest* must be page-locked memory, see, e.g. :func:`pagelocked_empty`.

    New in 0.93.

.. function:: memcpy_dtod(dest, src, size)
.. function:: memcpy_dtod_async(dest, src, size, stream=None)

    CUDA 3.0 and above.

    .. versionadded:: 0.94

.. function:: memcpy_peer(dest, src, size, dest_context=None, src_context=None)
.. function:: memcpy_peer_async(dest, src, size, dest_context=None, src_context=None, stream=None)

    CUDA 4.0 and above.

    .. versionadded:: 2011.1

.. function:: memcpy_dtoa(ary, index, src, len)
.. function:: memcpy_atod(dest, ary, index, len)
.. function:: memcpy_htoa(ary, index, src)
.. function:: memcpy_atoh(dest, ary, index)
.. function:: memcpy_atoa(dest, dest_index, src, src_index, len)

Structured Memory Transfers
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. class:: Memcpy2D()

    .. attribute:: src_x_in_bytes

        X Offset of the origin of the copy. (initialized to 0)

    .. attribute:: src_y

        Y offset of the origin of the copy. (initialized to 0)

    .. attribute:: src_pitch

        Size of a row in bytes at the origin of the copy.

    .. method:: set_src_host(buffer)

        Set the *buffer*, which must be a Python object adhering to the buffer interface,
        to be the origin of the copy.

    .. method:: set_src_array(array)

        Set the :class:`Array` *array* to be the origin of the copy.

    .. method:: set_src_device(devptr)

        Set the device address *devptr* (an :class:`int` or a
        :class:`DeviceAllocation`) as the origin of the copy.

    .. method:: set_src_unified(buffer)

        Same as :meth:`set_src_host`, except that *buffer* may also correspond
        to device memory.

        CUDA 4.0 and above. Requires unified addressing.

        .. versionadded:: 2011.1

    .. attribute :: dst_x_in_bytes

        X offset of the destination of the copy. (initialized to 0)

    .. attribute :: dst_y

        Y offset of the destination of the copy. (initialized to 0)

    .. attribute :: dst_pitch

        Size of a row in bytes at the destination of the copy.

    .. method:: set_dst_host(buffer)

        Set the *buffer*, which must be a Python object adhering to the buffer interface,
        to be the destination of the copy.

    .. method:: set_dst_array(array)

        Set the :class:`Array` *array* to be the destination of the copy.

    .. method:: set_dst_device(devptr)

        Set the device address *devptr* (an :class:`int` or a
        :class:`DeviceAllocation`) as the destination of the copy.

    .. method:: set_dst_unified(buffer)

        Same as :meth:`set_dst_host`, except that *buffer* may also correspond
        to device memory.

        CUDA 4.0 and above. Requires unified addressing.

        .. versionadded:: 2011.1

    .. attribute:: width_in_bytes

        Number of bytes to copy for each row in the transfer.

    .. attribute:: height

        Number of rows to copy.

    .. method:: __call__([aligned=True])

        Perform the specified memory copy, waiting for it to finish.
        If *aligned* is *False*, tolerate device-side misalignment
        for device-to-device copies that may lead to loss of
        copy bandwidth.

    .. method:: __call__(stream)

        Perform the memory copy asynchronously, serialized via the :class:`Stream`
        *stream*. Any host memory involved in the transfer must be page-locked.


.. class:: Memcpy3D()

    :class:`Memcpy3D` has the same members as :class:`Memcpy2D`, and additionally
    all of the following:

    .. attribute:: src_height

        Ignored when source is an :class:`Array`. May be 0 if Depth==1.

    .. attribute:: src_z

        Z offset of the origin of the copy. (initialized to 0)

    .. attribute:: dst_height

        Ignored when destination is an :class:`Array`. May be 0 if Depth==1.

    .. attribute:: dst_z

        Z offset of the destination of the copy. (initialized to 0)

    .. attribute:: depth

    :class:`Memcpy3D` is supported on CUDA 2.0 and above only.

.. class:: Memcpy3DPeer()

    :class:`Memcpy3DPeer` has the same members as :class:`Memcpy3D`,
    and additionally all of the following:

    .. method:: set_src_context(ctx)

    .. method:: set_dst_context(ctx)

    CUDA 4.0 and newer.

    .. versionadded:: 2011.1


Code on the Device: Modules and Functions
-----------------------------------------

.. class:: Module

    Handle to a CUBIN module loaded onto the device. Can be created with
    :func:`module_from_file` and :func:`module_from_buffer`.

    .. method:: get_function(name)

        Return the :class:`Function` *name* in this module.

        .. warning::

            While you can obtain different handles to the same function using this
            method, these handles all share the same state that is set through
            the ``set_XXX`` methods of :class:`Function`. This means that you
            can't obtain two different handles to the same function and
            :meth:`Function.prepare` them in two different ways.

    .. method:: get_global(name)

        Return a tuple `(device_ptr, size_in_bytes)` giving the device address
        and size of the global *name*.

        The main use of this method is to find the address of pre-declared
        `__constant__` arrays so they can be filled from the host before kernel
        invocation.

    .. method:: get_texref(name)

        Return the :class:`TextureReference` *name* from this module.

    .. method:: get_surfref(name)

        Return the :class:`SurfaceReference` *name* from this module.

        CUDA 3.1 and above.

        .. versionadded:: 0.94

.. function:: module_from_file(filename)

    Create a :class:`Module` by loading the CUBIN file *filename*.

.. function:: module_from_buffer(buffer, options=[], message_handler=None)

    Create a :class:`Module` by loading a PTX or CUBIN module from
    *buffer*, which must support the Python buffer interface.
    (For example, :class:`str` and :class:`numpy.ndarray` do.)

    :param options: A list of tuples (:class:`jit_option`, value).
    :param message_handler: A callable that is called with a
      arguments of ``(compile_success_bool, info_str, error_str)``
      which allows the user to process error and warning messages
      from the PTX compiler.

    Loading PTX modules as well as non-default values of *options* and
    *message_handler* are only allowed on CUDA 2.1 and newer.

.. class:: Function

    Handle to a *__global__* function in a :class:`Module`. Create using
    :meth:`Module.get_function`.

    .. method:: __call__(arg1, ..., argn, block=block_size, [grid=(1,1), [stream=None, [shared=0, [texrefs=[], [time_kernel=False]]]]])

        Launch *self*, with a thread block size of *block*. *block* must be a 3-tuple
        of integers.

        *arg1* through *argn* are the positional C arguments to the kernel. See
        :meth:`param_set` for details. See especially the warnings there.

        *grid* specifies, as a 2-tuple, the number of thread blocks to launch, as a
        two-dimensional grid.
        *stream*, if specified, is a :class:`Stream` instance serializing the
        copying of input arguments (if any), execution, and the copying
        of output arguments (again, if any).
        *shared* gives the number of bytes available to the kernel in
        *extern __shared__* arrays.
        *texrefs* is a :class:`list` of :class:`TextureReference` instances
        that the function will have access to.

        The function returns either *None* or the number of seconds spent
        executing the kernel, depending on whether *time_kernel* is *True*.

        This is a convenience interface that can be used instead of the
        :meth:`param_*` and :meth:`launch_*` methods below.  For a faster (but
        mildly less convenient) way of invoking kernels, see :meth:`prepare` and
        :meth:`prepared_call`.

        *arg1* through *argn* are allowed to be of the following types:

        * Subclasses of :class:`numpy.number`. These are sized number types
          such as :class:`numpy.uint32` or :class:`numpy.float32`.

        * :class:`DeviceAllocation` instances, which will become a device pointer
          to the allocated memory.

        * Instances of :class:`ArgumentHandler` subclasses. These can be used to
          automatically transfer :mod:`numpy` arrays onto and off of the device.

        * Objects supporting the Python :class:`buffer` interface. These chunks
          of bytes will be copied into the parameter space verbatim.

        * :class:`GPUArray` instances.

        .. warning::

            You cannot pass values of Python's native :class:`int` or :class:`float`
            types to param_set. Since there is no unambiguous way to guess the size
            of these integers or floats, it complains with a :exc:`TypeError`.

        .. note::

            This method has to guess the types of the arguments passed to it,
            which can make it somewhat slow. For a kernel that is invoked often,
            this can be inconvenient. For a faster (but mildly less convenient) way
            of invoking kernels, see :meth:`prepare` and :meth:`prepared_call`.

    .. method:: param_set_texref(texref)

        Make the :class:`TextureReference` texref available to the function.

    .. method:: prepare(arg_types, shared=None, texrefs=[])

        Prepare the invocation of this function by

        * setting up the argument types as `arg_types`. `arg_types` is expected
          to be an iterable containing type characters understood by the
          :mod:`struct` module or :class:`numpy.dtype` objects.

          (In addition, PyCUDA understands *'F'* and *'D'* for single- and
          double precision floating point numbers.)

        * Registering the texture references `texrefs` for use with this functions.
          The :class:`TextureReference` objects in `texrefs` will be retained,
          and whatever these references are bound to at invocation time will
          be available through the corresponding texture references within the
          kernel.

        Return `self`.

    .. method:: prepared_call(grid, block, *args, shared_size=0)

        Invoke `self` using :meth:`launch_grid`, with `args` a grid size of `grid`,
        and a block size of *block*.
        Assumes that :meth:`prepare` was called on *self*.
        The texture references given to :meth:`prepare` are set up as parameters, as
        well.

        .. versionchanged:: 2012.1
            *shared_size* was added.

    .. method:: prepared_timed_call(grid, block, *args, shared_size=0)

        Invoke `self` using :meth:`launch_grid`, with `args`, a grid size of `grid`,
        and a block size of *block*.
        Assumes that :meth:`prepare` was called on *self*.
        The texture references given to :meth:`prepare` are set up as parameters, as
        well.

        Return a 0-ary callable that can be used to query the GPU time consumed by
        the call, in seconds. Once called, this callable will block until
        completion of the invocation.

        .. versionchanged:: 2012.1
            *shared_size* was added.

    .. method:: prepared_async_call(grid, block, stream, *args, shared_size=0)

        Invoke `self` using :meth:`launch_grid_async`, with `args`, a grid size
        of `grid`, and a block size of *block*, serialized into the
        :class:`pycuda.driver.Stream` `stream`.  If `stream` is None, do the
        same as :meth:`prepared_call`.  Assumes that :meth:`prepare` was called
        on *self*.  The texture references given to :meth:`prepare` are set up
        as parameters, as well.

        .. versionchanged:: 2012.1
            *shared_size* was added.

    .. method:: get_attribute(attr)

        Return one of the attributes given by the
        :class:`function_attribute` value *attr*.

        All :class:`function_attribute` values may also be directly read
        as (lower-case) attributes on the :class:`Function` object itself,
        e.g. `func.num_regs`.

        CUDA 2.2 and newer.

        .. versionadded:: 0.93

    .. attribute:: set_cache_config(fc)

        See :class:`func_cache` for possible values of *fc*.

        CUDA 3.0 (post-beta) and newer.

        .. versionadded:: 0.94

    .. attribute:: set_shared_config(sc)

        See :class:`shared_config` for possible values of *sc*.

        CUDA 4.2 and newer.

        .. versionadded:: 2013.1

    .. attribute:: local_size_bytes

        The number of bytes of local memory used by this function.

        On CUDA 2.1 and below, this is only available if this function is part
        of a :class:`pycuda.compiler.SourceModule`.  It replaces the now-deprecated attribute
        `lmem`.

    .. attribute:: shared_size_bytes

        The number of bytes of shared memory used by this function.

        On CUDA 2.1 and below, this is only available if this function is part
        of a :class:`pycuda.compiler.SourceModule`.  It replaces the now-deprecated attribute
        `smem`.

    .. attribute:: num_regs

        The number of 32-bit registers used by this function.

        On CUDA 2.1 and below, this is only available if this function is part
        of a :class:`pycuda.compiler.SourceModule`.  It replaces the now-deprecated attribute
        `registers`.

    .. method:: set_shared_size(bytes)

        Set *shared* to be the number of bytes available to the kernel in
        *extern __shared__* arrays.

        .. warning:: Deprecated as of version 2011.1.

    .. method:: set_block_shape(x, y, z)

        Set the thread block shape for this function.

        .. warning:: Deprecated as of version 2011.1.

    .. method:: param_set(arg1, ... argn)

        Set the thread block shape for this function.

        .. warning:: Deprecated as of version 2011.1.

    .. method:: param_set_size(bytes)

        Size the parameter space to *bytes*.

        .. warning:: Deprecated as of version 2011.1.

    .. method:: param_seti(offset, value)

        Set the integer at *offset* in the parameter space to *value*.

        .. warning:: Deprecated as of version 2011.1.

    .. method:: param_setf(offset, value)

        Set the float at *offset* in the parameter space to *value*.

        .. warning:: Deprecated as of version 2011.1.

    .. method:: launch()

        Launch a single thread block of *self*.

        .. warning:: Deprecated as of version 2011.1.

    .. method:: launch_grid(width, height)

        Launch a width*height grid of thread blocks of *self*.

        .. warning:: Deprecated as of version 2011.1.

    .. method:: launch_grid_async(width, height, stream)

        Launch a width*height grid of thread blocks of *self*, sequenced
        by the :class:`Stream` *stream*.

        .. warning:: Deprecated as of version 2011.1.


.. class:: ArgumentHandler(array)

.. class:: In(array)

    Inherits from :class:`ArgumentHandler`. Indicates that :class:`buffer`
    *array* should be copied to the compute device before invoking the kernel.

.. class:: Out(array)

    Inherits from :class:`ArgumentHandler`. Indicates that :class:`buffer`
    *array* should be copied off the compute device after invoking the kernel.

.. class:: InOut(array)

    Inherits from :class:`ArgumentHandler`. Indicates that :class:`buffer`
    *array* should be copied both onto the compute device before invoking
    the kernel, and off it afterwards.

Profiler Control
================

CUDA 4.0 and newer.


.. function:: initialize_profiler(config_file, output_file, output_mode)

    *output_mode* is one of the attributes of :class:`profiler_output_mode`.

    .. versionadded:: 2011.1

.. function:: start_profiler()

    .. versionadded:: 2011.1

.. function:: stop()

    .. versionadded:: 2011.1

Just-in-time Compilation
========================

.. module:: pycuda.compiler

.. data:: DEFAULT_NVCC_FLAGS

    .. versionadded:: 2011.1

    If no *options* are given in the calls below, the value of this list-type
    variable is used instead. This may be useful for injecting necessary flags
    into the compilation of automatically compiled kernels, such as those used
    by the module :mod:`pycuda.gpuarray`.

    The initial value of this variable is taken from the environment variable
    :envvar:`PYCUDA_DEFAULT_NVCC_FLAGS`.

    If you modify this variable in your code, please be aware that this is a
    globally shared variable that may be modified by multiple packages. Please
    exercise caution in such modifications--you risk breaking other people's
    code.

.. class:: SourceModule(source, nvcc="nvcc", options=None, keep=False, no_extern_c=False, arch=None, code=None, cache_dir=None, include_dirs=[])

    Create a :class:`Module` from the CUDA source code *source*. The Nvidia
    compiler *nvcc* is assumed to be on the :envvar:`PATH` if no path to it is
    specified, and is invoked with *options* to compile the code. If *keep* is
    *True*, the compiler output directory is kept, and a line indicating its
    location in the file system is printed for debugging purposes.

    Unless *no_extern_c* is *True*, the given source code is wrapped in
    *extern "C" { ... }* to prevent C++ name mangling.

    `arch` and `code` specify the values to be passed for the :option:`-arch`
    and :option:`-code` options on the :program:`nvcc` command line. If `arch` is
    `None`, it defaults to the current context's device's compute capability.
    If `code` is `None`, it will not be specified.

    `cache_dir` gives the directory used for compiler caching.  If `None`
    then `cache_dir` is taken to be :envvar:`PYCUDA_CACHE_DIR` if set or
    a sensible per-user default.  If passed as `False`, caching is disabled.

    If the environment variable :envvar:`PYCUDA_DISABLE_CACHE` is set to
    any value then caching is disabled.  This preference overrides any
    value of `cache_dir` and can be used to disable caching globally.

    This class exhibits the same public interface as :class:`pycuda.driver.Module`, but
    does not inherit from it.

    *Change note:* :class:`SourceModule` was moved from :mod:`pycuda.driver` to
    :mod:`pycuda.compiler` in version 0.93.

.. function:: compile(source, nvcc="nvcc", options=None, keep=False,
        no_extern_c=False, arch=None, code=None, cache_dir=None,
        include_dirs=[])

    Perform the same compilation as the corresponding
    :class:`SourceModule` constructor, but only return
    resulting *cubin* file as a string. In particular,
    do not upload the code to the GPU.
