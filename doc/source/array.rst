The :class:`GPUArray` Array Class
=================================

.. module:: pycuda.gpuarray

.. class:: GPUArray(shape, dtype, stream=None)

  A :class:`numpy.ndarray` work-alike that stores its data and performs its
  computations on the compute device.  *shape* and *dtype* work exactly as in
  :mod:`numpy`.  Arithmetic methods in :class:`GPUArray` support the
  broadcasting of scalars. (e.g. `array+5`) If the
  :class:`pycuda.driver.Stream` *stream* is specified, all computations on
  *self* are sequenced into it.

  .. attribute :: gpudata
    
    The :class:`pycuda.driver.DeviceAllocation` instance created for the memory that backs
    this :class:`GPUArray`.

  .. attribute :: shape

    The tuple of lengths of each dimension in the array.

  .. attribute :: dtype 
    
    The numpy :class:`numpy.dtype` of the items in the GPU array.
    
  .. attribute :: size
    
    The number of meaningful entries in the array. Can also be computed by
    multiplying up the numbers in :attr:`shape`.

  .. attribute :: mem_size
    
    The total number of entries, including padding, that are present in
    the array. Padding may arise for example because of pitch adjustment by 
    :func:`pycuda.driver.mem_alloc_pitch`.

  .. attribute :: nbytes
    
    The size of the entire array in bytes. Computed as :attr:`size` times 
    ``dtype.itemsize``.

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

  .. method :: mul_add(self, selffac, other, otherfac, add_timer=None):
    
    Return `selffac*self + otherfac*other`. *add_timer*, if given, 
    is invoked with the result from 
    :meth:`pycuda.driver.Function.prepared_timed_call`.

  .. method :: __add__(other)
  .. method :: __sub__(other)
  .. method :: __iadd__(other)
  .. method :: __isub__(other)
  .. method :: __neg__(other)
  .. method :: __mul__(other)
  .. method :: __div__(other)
  .. method :: __rdiv__(other)
  .. method :: __pow__(other)

  .. method :: __abs__()

    Return a :class:`GPUArray` containing the absolute value of each
    element of *self*.

  .. UNDOC reverse()
  
  .. method :: fill(scalar)

    Fill the array with *scalar*.

  .. method:: bind_to_texref(texref)

    Bind *self* to the :class:`TextureReference` *texref*.
    
Constructing :class:`GPUArray` Instances
----------------------------------------

.. function:: to_gpu(ary, stream=None)
  
  Return a :class:`GPUArray` that is an exact copy of the :class:`numpy.ndarray`
  instance *ary*. Optionally sequence on *stream*.
  
.. function:: empty(shape, dtype, stream)

  A synonym for the :class:`GPUArray` constructor.

.. function:: zeros(shape, dtype, stream)

  Same as :func:`empty`, but the :class:`GPUArray` is zero-initialized before
  being returned.

.. function:: empty_like(other_ary)

  Make a new, uninitialized :class:`GPUArray` having the same properties 
  as *other_ary*.

.. function:: zeros_like(other_ary)

  Make a new, zero-initialized :class:`GPUArray` having the same properties
  as *other_ary*.

.. function:: arange(start, stop, step, dtype=numpy.float32)

  Create a :class:`GPUArray` filled with numbers spaced `step` apart,
  starting from `start` and ending at `stop`.
  
  For floating point arguments, the length of the result is
  `ceil((stop - start)/step)`.  This rule may result in the last
  element of the result being greater than `stop`.

Elementwise Functions on :class:`GPUArrray` Instances
-----------------------------------------------------

.. module:: pycuda.cumath

The :mod:`pycuda.cumath` module contains elementwise 
workalikes for the functions contained in :mod:`math`.

Rounding and Absolute Value
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: fabs(array)
.. function:: ceil(array)
.. function:: floor(array)

General Transcendental Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: exp(array)
.. function:: log(array)
.. function:: log10(array)
.. function:: sqrt(array)

Trigonometric Functions
^^^^^^^^^^^^^^^^^^^^^^^

.. function:: sin(array)
.. function:: cos(array)
.. function:: tan(array)
.. function:: asin(array)
.. function:: acos(array)
.. function:: atan(array)

Hyperbolic Functions
^^^^^^^^^^^^^^^^^^^^

.. function:: sinh(array)
.. function:: cosh(array)
.. function:: tanh(array)

Floating Point Decomposition and Assembly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: fmod(arg, mod)

    Return the floating point remainder of the division `arg/mod`,
    for each element in `arg` and `mod`.

.. function:: frexp(arg)

    Return a tuple `(significands, exponents)` such that 
    `arg == significand * 2**exponent`.
    
.. function:: ldexp(significand, exponent)

    Return a new array of floating point values composed from the
    entries of `significand` and `exponent`, paired together as
    `result = significand * 2**exponent`.
        
.. function:: modf(arg)

    Return a tuple `(fracpart, intpart)` of arrays containing the
    integer and fractional parts of `arg`. 

Generating Arrays of Random Numbers
-----------------------------------

.. module:: pycuda.curandom

.. function:: rand(shape, dtype=numpy.float32)

  Return an array of `shape` filled with random values of `dtype`
  in the range [0,1).

Single-pass Expression Evaluation
---------------------------------

.. warning::

  The following functionality is included in this documentation in the 
  hope that it may be useful, but its interface may change in future
  revisions. Feedback is welcome.

.. module:: pycuda.elementwise

Evaluating involved expressions on :class:`GPUArray` instances can be
somewhat inefficient, because a new temporary is created for each 
intermediate result. The functionality in the module :mod:`pycuda.elementwise`
contains tools to help generate kernels that evaluate multi-stage expressions
on one or several operands in a single pass.

.. class:: ElementwiseKernel(arguments, operation, name="kernel", keep=False, options=[])

    Generate a kernel that takes a number of scalar or vector *arguments*
    and performs the scalar *operation* on each entry of its arguments, if that 
    argument is a vector.

    *arguments* is specified as a string formatted as a C argument list. 
    *operation* is specified as a C assignment statement, without a semicolon. 
    Vectors in *operation* should be indexed by the variable *i*.

    *name* specifies the name as which the kernel is compiled, *keep*
    and *options* are passed unmodified to :class:`pycuda.driver.SourceModule`.

    .. method:: __call__(*args)

        Invoke the generated scalar kernel. The arguments may either be scalars or
        :class:`GPUArray` instances.

Here's a usage example::

    import pycuda.gpuarray as gpuarray
    import pycuda.driver as cuda
    import pycuda.autoinit
    import numpy
    from pycuda.curandom import rand as curand

    a_gpu = curand((50,))
    b_gpu = curand((50,))

    from pycuda.elementwise import ElementwiseKernel
    lin_comb = ElementwiseKernel(
            "float a, float *x, float b, float *y, float *z",
            "z[i] = a*x[i] + b*y[i]",
            "linear_combination")

    c_gpu = gpuarray.empty_like(a_gpu)
    lin_comb(5, a_gpu, 6, b_gpu, c_gpu)

    import numpy.linalg as la
    assert la.norm((c_gpu - (5*a_gpu+6*b_gpu)).get()) < 1e-5

(You can find this example as :file:`examples/demo_elementwise.py` in the PyCuda 
distribution.)
