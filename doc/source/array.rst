The :class:`GPUArray` Array Class
=================================

.. module:: pycuda.gpuarray

.. class:: GPUArray(shape, dtype, stream=None)

  A :class:`numpy.ndarray` work-alike that stores its data and
  performs its computations on the compute device.

  *shape* and *dtype* work exactly as in :mod:`numpy`.

  If *stream* is specified, all computations on *self* are 
  sequenced into it.

  .. method :: set(ary, stream=None)

    Transfer the contents the :class:`numpy.ndarray` object *ary*
    onto the device, optionally sequenced on *stream*.

    *ary* must have the same dtype and size (not necessarily shape) as *self*.

  .. method :: get(ary=None, stream=None, pagelocked=False)

    Transfer the contents of *self* into *ary* or a newly allocated
    :mod:`numpy.ndarray`. If *ary* is given, it must have the right
    size (not necessarily shape) and dtype. If it is not given,
    *pagelocked* specifies whether the new array is allocated 
    page-locked.

  .. method :: __add__(other)
  .. method :: __sub__(other)
  .. method :: __iadd__(other)
  .. method :: __isub__(other)
  .. method :: __neg__(other)
  .. method :: __mul__(other)

    *other* may be a scalar or another :class:`GPUArray`.
    
  .. method :: __imul__(other)

    *other* may be a scalar or another :class:`GPUArray`.

  .. method :: fill(scalar)

    Fill the array with *scalar*.

  .. method:: bind_to_texref(texref)

    Bind *self* to the :class:`TextureReference` *texref*.
    
.. function:: to_gpu(ary, stream=None)
  
  Return a :class:`GPUArray` that is an exact copy of the :class:`numpy.ndarray`
  instance *ary*. Optionally sequence on *stream*.
  
.. function:: empty(shape, dtype, stream)

  A synonym for the :class:`GPUArray` constructor.

.. function:: zeros(shape, dtype, stream)

  Same as :func:`empty`, but the :class:`GPUArray` is zero-initialized before
  being returned.
