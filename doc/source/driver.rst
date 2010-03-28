.. _reference-doc:

Device Interface Reference Documentation
========================================

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

    Thrown when :class:`SourceModule` compilation fails.

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

    .. attribute:: BLOCKING_SYNC

        Use blocking synchronization. CUDA 2.2 and newer.

    .. attribute:: MAP_HOST

        Support mapped pinned allocations. CUDA 2.2 and newer.

    .. attribute:: FLAGS_MASK

        Mask of valid flags in this bitfield.


.. class:: event_flags

    Flags for :class:`Event`. CUDA 2.2 and newer.

    .. attribute:: DEFAULT
    .. attribute:: BLOCKING_SYNC

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

        CUDA 3.0 and above

        .. versionadded:: 0.94

    .. attribute:: SURFACE_ALIGNMENT

        CUDA 3.0 (post-beta) and above

        .. versionadded:: 0.94

    .. attribute:: CONCURRENT_KERNELS

        CUDA 3.0 (post-beta) and above

        .. versionadded:: 0.94

    .. attribute:: ECC_ENABLED

        CUDA 3.0 (post-beta) and above

        .. versionadded:: 0.94

.. class:: function_attribute

    Flags for :meth:`Function.get_attribute`. CUDA 2.2 and newer.

    .. attribute:: MAX_THREADS_PER_BLOCK
    .. attribute:: SHARED_SIZE_BYTES
    .. attribute:: CONST_SIZE_BYTES
    .. attribute:: LOCAL_SIZE_BYTES
    .. attribute:: NUM_REGS
    .. attribute:: PTX_VERSION

        CUDA 3.0 (post-beta) and above

        .. versionadded:: 0.94

    .. attribute:: BINARY_VERSION

        CUDA 3.0 (post-beta) and above

        .. versionadded:: 0.94

    .. attribute:: MAX

.. class:: func_cache

    See :meth:`Function.set_cache_config`. CUDA 3.0 (post-beta) and above

    .. versionadded:: 0.94

    .. attribute:: PREFER_NONE
    .. attribute:: PREFER_SHARED
    .. attribute:: PREFER_L1

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

    .. attribute ARRAY3D_2DARRAY

        CUDA 3.0 and above

        .. versionadded:: 0.94

.. class:: address_mode

    .. attribute:: WRAP
    .. attribute:: CLAMP
    .. attribute:: MIRROR

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
    .. attribute:: EXCLUSIVE
    .. attribute:: PROHIBITED

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

        CUDA 3.0 and above

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

    A handle to the *number*'th CUDA device. See also :mod:`pycuda.autoinit`.

    .. staticmethod:: count()

        Return the number of CUDA devices found.

    .. method:: name()

        Return the name of this CUDA device.

    .. method:: compute_cabability()

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

Concurrency and Streams
-----------------------

.. class:: Stream(flags=0)

    A handle for a queue of operations that will be carried out in order.

    .. method:: synchronize()

        Wait for all activity on this stream to cease, then return.

    .. method:: is_done()

        Return *True* iff all queued operations have completed.

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

.. _pagelocked_memory :

Pagelocked Host Memory
^^^^^^^^^^^^^^^^^^^^^^

.. function:: pagelocked_empty(shape, dtype, order="C", mem_flags=0)

    Allocate a pagelocked :class:`numpy.ndarray` of *shape*, *dtype* and *order*.

    *mem_flags* may be one of the values in :class:`host_alloc_flags`.
    It may only be non-zero on CUDA 2.2 and newer.

    For the meaning of the other parameters, please refer to the :mod:`numpy`
    documentation.

.. function:: pagelocked_zeros(shape, dtype, order="C", mem_flags=0)

    Allocate a pagelocked :class:`numpy.ndarray` of *shape*, *dtype* and *order* that
    is zero-initialized.

    *mem_flags* may be one of the values in :class:`host_alloc_flags`.
    It may only be non-zero on CUDA 2.2 and newer.

    For the meaning of the other parameters, please refer to the :mod:`numpy`
    documentation.

.. function:: pagelocked_empty_like(array, mem_flags=0)

    Allocate a pagelocked :class:`numpy.ndarray` with the same shape, dtype and order
    as *array*.

    *mem_flags* may be one of the values in :class:`host_alloc_flags`.
    It may only be non-zero on CUDA 2.2 and newer.

.. function:: pagelocked_zeros_like(array, mem_flags=0)

    Allocate a pagelocked :class:`numpy.ndarray` with the same shape, dtype and order
    as *array*. Initialize it to 0.

    *mem_flags* may be one of the values in :class:`host_alloc_flags`.
    It may only be non-zero on CUDA 2.2 and newer.

The :class:`numpy.ndarray` instances returned by these functions
have an attribute *base* that references an object of type

.. class:: HostAllocation

    An object representing an allocation of pagelocked
    host memory.  Once this object is deleted, its associated
    device memory is freed.

    .. method:: free()

        Release the held memory now instead of when this object
        becomes unreachable. Any further use of the object (or its
        associated :mod:`numpy` array) is an error
        and will lead to undefined behavior.

    .. method:: get_device_pointer()

        Return a device pointer that indicates the address at which
        this memory is mapped into the device's address space.

        Only available on CUDA 2.2 and newer.

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

    New in 0.93.

.. function:: memcpy_dtoh(dest, src)

    Copy from the device pointer *src* (an :class:`int` or a
    :class:`DeviceAllocation`) to the Python buffer *dest*. The size of the copy
    is determined by the size of the buffer.

    Optionally execute asynchronously, serialized via *stream*. In
    this case, *dest* must be page-locked.

.. function:: memcpy_dtoh_async(dest, src, stream=None)

    Copy from the device pointer *src* (an :class:`int` or a
    :class:`DeviceAllocation`) to the Python buffer *dest* asynchronously,
    optionally serialized via *stream*. The size of the copy
    is determined by the size of the buffer.

.. function:: memcpy_dtod(dest, src, size)
.. function:: memcpy_dtod_async(dest, src, size, stream=None)

    CUDA 3.0 and above

    .. versionadded:: 0.94

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

    .. method:: param_set(arg1, ... argn)

        Set up *arg1* through *argn* as positional C arguments to *self*. They are
        allowed to be of the following types:

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

    .. method:: set_block_shape(x, y, z)

        Set the thread block shape for this function.

    .. method:: set_shared_size(bytes)

        Set *shared* to be the number of bytes available to the kernel in
        *extern __shared__* arrays.

    .. method:: param_set_size(bytes)

        Size the parameter space to *bytes*.

    .. method:: param_seti(offset, value)

        Set the integer at *offset* in the parameter space to *value*.

    .. method:: param_setf(offset, value)

        Set the float at *offset* in the parameter space to *value*.

    .. method:: param_set_texref(texref)

        Make the :class:`TextureReference` texref available to the function.

    .. method:: launch()

        Launch a single thread block of *self*.

    .. method:: launch_grid(width, height)

        Launch a width*height grid of thread blocks of *self*.

    .. method:: launch_grid_async(width, height, stream)

        Launch a width*height grid of thread blocks of *self*, sequenced
        by the :class:`Stream` *stream*.

    .. method:: prepare(arg_types, block, shared=None, texrefs=[])

        Prepare the invocation of this function by

        * setting up the argument types as `arg_types`. `arg_types` is expected
          to be an iterable containing type characters understood by the
          :mod:`struct` module or :class:`numpy.dtype` objects.

          (In addition, PyCUDA understands *'F'* and *'D'* for single- and
          double precision floating point numbers.)

        * setting the thread block shape for this function to `block`.

        * Registering the texture references `texrefs` for use with this functions.
          The :class:`TextureReference` objects in `texrefs` will be retained,
          and whatever these references are bound to at invocation time will
          be available through the corresponding texture references within the
          kernel.

        Return `self`.

    .. method:: prepared_call(grid, *args)

        Invoke `self` using :meth:`launch_grid`, with `args` and a grid size of `grid`.
        Assumes that :meth:`prepare` was called on *self*.
        The texture references given to :meth:`prepare` are set up as parameters, as
        well.

    .. method:: prepared_timed_call(grid, *args)

        Invoke `self` using :meth:`launch_grid`, with `args` and a grid size of `grid`.
        Assumes that :meth:`prepare` was called on *self*.
        The texture references given to :meth:`prepare` are set up as parameters, as
        well.

        Return a 0-ary callable that can be used to query the GPU time consumed by
        the call, in seconds. Once called, this callable will block until
        completion of the invocation.

    .. method:: prepared_async_call(grid, stream, *args)

        Invoke `self` using :meth:`launch_grid_async`, with `args` and a grid
        size of `grid`, serialized into the :class:`pycuda.driver.Stream` `stream`.
        If `stream` is None, do the same as :meth:`prepared_call`.
        Assumes that :meth:`prepare` was called on *self*.
        The texture references given to :meth:`prepare` are set up as parameters, as
        well.

    .. method:: get_attribute(attr)

        Return one of the attributes given by the
        :class:`function_attribute` value *attr*.

        All :class:`function_attribute` values may also be directly read
        as (lower-case) attributes on the :class:`Function` object itself,
        e.g. `func.num_regs`.

        CUDA 2.2 and newer.

        .. versionadded:: 0.93

    .. attribute:: set_cache_config(fc)

        CUDA 3.0 (post-beta) and newer.

        .. versionadded:: 0.94

    .. attribute:: local_size_bytes

        The number of bytes of local memory used by this function.

        On CUDA 2.1 and below, this is only available if this function is part
        of a :class:`SourceModule`.  It replaces the now-deprecated attribute
        `lmem`.

    .. attribute:: shared_size_bytes

        The number of bytes of shared memory used by this function.

        On CUDA 2.1 and below, this is only available if this function is part
        of a :class:`SourceModule`.  It replaces the now-deprecated attribute
        `smem`.

    .. attribute:: num_regs

        The number of 32-bit registers used by this function.

        On CUDA 2.1 and below, this is only available if this function is part
        of a :class:`SourceModule`.  It replaces the now-deprecated attribute
        `registers`.

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

Just-in-time Compilation
========================

.. module:: pycuda.compiler

.. class:: SourceModule(source, nvcc="nvcc", options=[], keep=False, no_extern_c=False, arch=None, code=None, cache_dir=None)

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

    `cache_dir` gives the directory used for compiler caching. It has a
    sensible per-user default. If it is set to `False`, caching is
    disabled.

    This class exhibits the same public interface as :class:`Module`, but
    does not inherit from it.

    *Change note:* :class:`SourceModule` was moved from :mod:`pycuda.driver` to
    :mod:`pycuda.compiler` in version 0.93.

.. function:: compile(source, nvcc="nvcc", options=[], keep=False,
        no_extern_c=False, arch=None, code=None, cache_dir=None,
        include_dirs=[])

    Perform the same compilation as the corresponding 
    :class:`SourceModule` constructor, but only return
    resulting *cubin* file as a string. In particular,
    do not upload the code to the GPU.
