.. _reference-doc:

Device Interface Reference Documentation
========================================

.. module:: pycuda.driver
   :synopsis: Use CUDA devices from Python

.. moduleauthor:: Andreas Kloeckner <inform@tiker.net>

Constants
---------

.. class:: ctx_flags

  .. attribute:: SCHED_AUTO
  .. attribute:: SCHED_SPIN
  .. attribute:: SCHED_YIELD
  .. attribute:: SCHED_MASK
  .. attribute:: SCHED_FLAGS_MASK

.. class:: device_attribute

  .. attribute:: MAX_THREADS_PER_BLOCK
  .. attribute:: MAX_BLOCK_DIM_X
  .. attribute:: MAX_BLOCK_DIM_Y
  .. attribute:: MAX_BLOCK_DIM_Z
  .. attribute:: MAX_GRID_DIM_X
  .. attribute:: MAX_GRID_DIM_Y
  .. attribute:: MAX_GRID_DIM_Z
  .. attribute:: MAX_SHARED_MEMORY_PER_BLOCK
  .. attribute:: SHARED_MEMORY_PER_BLOCK
  .. attribute:: TOTAL_CONSTANT_MEMORY
  .. attribute:: WARP_SIZE
  .. attribute:: MAX_PITCH
  .. attribute:: MAX_REGISTERS_PER_BLOCK
  .. attribute:: REGISTERS_PER_BLOCK
  .. attribute:: CLOCK_RATE
  .. attribute:: TEXTURE_ALIGNMENT
  .. attribute:: GPU_OVERLAP
  .. attribute:: MULTIPROCESSOR_COUNT

.. class:: array_format

  .. attribute:: UNSIGNED_INT8
  .. attribute:: UNSIGNED_INT16
  .. attribute:: UNSIGNED_INT32
  .. attribute:: SIGNED_INT8
  .. attribute:: SIGNED_INT16
  .. attribute:: SIGNED_INT32
  .. attribute:: HALF
  .. attribute:: FLOAT

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

Devices and Contexts
--------------------

.. function:: init(flags=0)

  Initialize CUDA. 
  
  .. warning:: This must be called before any other function in this module.

.. class:: Device(number)

  A handle to the *number*'th CUDA device.

  .. method:: count() [static method]

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

  .. method:: get_attributes()
    
    Return all device attributes in a :class:`dict`, with keys from
    :class:`device_attribute`.

  .. method:: make_context(flags=ctx_flags.SCHED_AUTO)
    
    Create a :class:`context` on this device, with flags taken from the
    :class:`ctx_flags` values.

    Also make the newly-created context the current context.


.. class:: Context
  
  An equivalent of a UNIX process on the compute device.
  Create instances of this class using :meth:`Device.make_context`.

  .. method:: detach()

    Decrease the reference count on this context. If the reference count
    hits zero, the context is deleted.

  .. method:: push()
    
    Make *self* the active context, pushing it on top of the context stack.

  .. method:: pop()

    Remove *self* from the top of the context stack, deactivating it.

  .. method:: get_device() [static method]

    Return the device that the current context is working on.

  .. method:: synchronize() [static method]

    Wait for all activity in this context to cease, then return.

Concurrency and Streams
-----------------------

.. class:: Stream(flags=0)
  
  A handle for a queue of operations that will be carried out in order.

  .. method:: synchronize()
    
    Wait for all activity on this stream to cease, then return.

  .. method:: is_done()

    Return *True* iff all queued operations have completed.

.. class:: Event(flags=0)

  .. method:: record()

    Insert a recording point for *self* into the global device execution
    stream.

  .. method:: record_in_stream(stream)

    Insert a recording point for *self* into the :class:`Stream` *stream*

  .. method:: synchronize()

    Wait until the device execution stream reaches this event.

  .. method:: query()

    Return *True* if the device execution stream has reached this event.

  .. method:: time_since(event)

    Return the time in milliseconds that has passed between *self* and *event*.

  .. method:: time_till(event)

    Return the time in milliseconds that has passed between *event* and *self*.


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
  
  Allocate enough memory for *buffer*, which adheres to the Python
  :class:`buffer` interface. Copy the contents of *buffer* onto the 
  device.

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


Pagelocked Host Memory
^^^^^^^^^^^^^^^^^^^^^^

.. function:: pagelocked_empty(shape, dtype, order="C")

  Allocate a pagelocked :mod:`numpy` array of *shape*, *dtype* and *order*.
  For the meaning of these parameters, please refer to the :mod:`numpy` 
  documentation.

.. function:: pagelocked_zeros(shape, dtype, order="C")

  Allocate a pagelocked :mod:`numpy` array of *shape*, *dtype* and *order* that
  is zero-initialized.

  For the meaning of these parameters, please refer to the :mod:`numpy` 
  documentation.

.. function:: pagelocked_empty_like(array)

  Allocate a pagelocked :mod:`numpy` array with the same shape, dtype and order
  as *array*.

.. function:: pagelocked_zeros_like(array)

  Allocate a pagelocked :mod:`numpy` array with the same shape, dtype and order
  as *array*. Initialize it to 0.

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

    A value of type :class:`array_format`.

  .. attribute:: num_channels

.. class:: Array(descriptor)

  A 2D or 3D memory block that can only be accessed via 
  texture references.

  *descriptor* can be of type :class:`ArrayDescriptor` or
  :class:`ArrayDescriptor3D`.

  .. method::  get_descriptor()

    Return a :class:`ArrayDescriptor` object for this 2D array, 
    like the one that was used to create it.

  .. method::  get_descriptor_3d()

    Return a :class:`ArrayDescriptor3D` object for this 3D array, 
    like the one that was used to create it.

.. class:: TextureReference()
  
  A handle to a binding of either linear memory or an :class:`Array` to
  a texture unit.

  .. method:: set_array(array)
  
    Bind *self* to the :class:`Array` *array*.

  .. method:: set_address(devptr, bytes)
  
    Bind *self* to the a chunk of linear memory starting at the integer address 
    *devptr*, encompassing a number of *bytes*.

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
  .. method:: get_flags()

.. data:: TRSA_OVERRIDE_FORMAT
.. data:: TRSF_READ_AS_INTEGER
.. data:: TRSF_NORMALIZED_COORDINATES
.. data:: TR_DEFAULT

.. function:: matrix_to_array(matrix)

  Turn the two-dimensional :class:`numpy.ndarray` object *matrix* into an
  :class:`Array`. The dimensions are in the same order as they should be used
  in the :cfunc:`tex2D` argument sequence.

.. function:: make_multichannel_2d_array(matrix)

  Turn the three-dimensional :class:`numpy.ndarray` object *matrix* into
  an 2D :class:`Array` with multiple channels, where the number of channels
  is the first dimension, and the remaining dimensions are in the same order
  as they should be used in the :cfunc:`tex2D` argument sequence.

Initializing Device Memory
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: memset_d8(dest, data, size)
.. function:: memset_d16(dest, data, size)
.. function:: memset_d32(dest, data, size)

.. function:: memset_d2d8(dest, pitch, data, width, height)
.. function:: memset_d2d16(dest, pitch, data, width, height)
.. function:: memset_d2d32(dest, pitch, data, width, height)

Unstructured Memory Transfers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: memcpy_htod(dest, src, stream=None)

  Copy from the Python buffer *src* to the device pointer *dest* 
  (an :class:`int` or a :class:`DeviceAllocation`). The size of
  the copy is determined by the size of the buffer. 
  
  Optionally execute asynchronously, serialized via *stream*. In
  this case, *src* must be page-locked.

.. function:: memcpy_dtoh(dest, src, stream=None)

  Copy from the device pointer *dest* (an :class:`int` or a
  :class:`DeviceAllocation`) to the Python buffer *src*. The size of the copy
  is determined by the size of the buffer.

  Optionally execute asynchronously, serialized via *stream*. In
  this case, *dest* must be page-locked.

.. function:: memcpy_dtod(dest, src, size)
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
    If *aligned* is *False*, tolerate misalignment that may lead
    to severe loss of copy bandwidth.

  .. method:: __call__(stream)

    Perform the memory copy asynchronously, serialized via the :class:`Stream`
    *stream*. Any host memory involved in the transfer must be page-locked.


.. class:: Memcpy3D()

  :class:`Memcpy3D` has the same members as :class:`Memcpy2D`, and additionally
  all of the following:

  .. attribute:: src_z
    
    Z offset of the origin of the copy. (initialized to 0)

  .. attribute:: src_lod
    
  .. attribute:: dst_z
    
    Z offset of the destination of the copy. (initialized to 0)

  .. attribute:: dst_lod

  .. attribute:: depth

Code on the Device: Modules and Functions
-----------------------------------------

.. class:: Module
  
  Handle to a CUBIN module loaded onto the device. Can be created with
  :func:`module_from_file` and :func:`module_from_buffer`.

  .. method:: get_function(name)
    
    Return the :class:`Function` *name* in this module.

  .. method:: get_global(name)

    Return the device address of the global *name* as an :class:`int`.

  .. method:: get_texref(name)

    Return the :class:`TextureReference` *name* from this module.

.. function:: module_from_file(filename)
  
  Create a :class:`Module` by loading the CUBIN file *filename*.

.. function:: module_from_buffer(buffer)

  Create a :class:`Module` by loading a CUBIN from *buffer*, which must
  support the Python buffer interface. (For example, :class:`str` and 
  :class:`numpy.ndarray` do.)
  

.. class:: Function

  Handle to a *__global__* function in a :class:`Module`. Create using
  :meth:`Module.get_function`.

  .. method:: __call__(arg1, ..., argn, block=block_size, [grid=(1,1), [stream=None, [shared=0, [texrefs=[], [time_kernel=False]]]]])

    Launch *self*, with a thread block size of *block*. *block* must be a 3-tuple
    of integers.

    *arg1* through *argn* are the positional C arguments to the kernel. See
    :meth:`param_set` for details.
    
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

    This is a convenience interface that replaces all the following functions.

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

.. class:: SourceModule(source, nvcc="nvcc", options=[], keep=False, 
  no_extern_c=False)
  
  Create a :class:`Module` from the CUDA source code *source*. The Nvidia
  compiler *nvcc* is assumed to be on the :envvar:`PATH` if no path to it is
  specified, and is invoked with *options* to compile the code. If *keep* is
  *True*, the compiler output directory is kept, and a line indicating its
  location in the file system is printed for debugging purposes.

  Unless *no_extern_c* is *True*, the given source code is wrapped in
  *extern "C" { ... }* to prevent C++ name mangling.

  This class exhibits the same public interface as :class:`Module`, but 
  does not inherit from it.
