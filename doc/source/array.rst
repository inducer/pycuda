The :class:`GPUArray` Array Class
=================================

.. module:: pycuda.gpuarray

.. class:: GPUArray(shape, dtype, allocator=None)

    A :class:`numpy.ndarray` work-alike that stores its data and performs its
    computations on the compute device.  *shape* and *dtype* work exactly as in
    :mod:`numpy`.  Arithmetic methods in :class:`GPUArray` support the
    broadcasting of scalars. (e.g. `array+5`) If the

    *allocator* is a callable that, upon being called with an argument of the number
    of bytes to be allocated, returns an object that can be cast to an
    :class:`int` representing the address of the newly allocated memory.
    Observe that both :func:`pycuda.driver.mem_alloc` and
    :meth:`pycuda.tools.DeviceMemoryPool.alloc` are a model of this interface.

    .. attribute :: gpudata

        The :class:`pycuda.driver.DeviceAllocation` instance created for the memory that backs
        this :class:`GPUArray`.

    .. attribute :: shape

        The tuple of lengths of each dimension in the array.

    .. attribute :: dtype

        The :class:`numpy.dtype` of the items in the GPU array.

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

    .. method :: __len__()

        Returns the size of the leading dimension of *self*.

      .. warning ::

        This method existed in version 0.93 and below, but it returned the value
        of :attr:`size` instead of its current value. The change was made in order
        to match :mod:`numpy`.

    .. method :: set(ary)

        Transfer the contents the :class:`numpy.ndarray` object *ary*
        onto the device.

        *ary* must have the same dtype and size (not necessarily shape) as *self*.

    .. method :: set_async(ary, stream=None)

        Asynchronously transfer the contents the :class:`numpy.ndarray` object *ary*
        onto the device, optionally sequenced on *stream*.

        *ary* must have the same dtype and size (not necessarily shape) as *self*.

    .. method :: get(ary=None, stream=None, pagelocked=False)

        Transfer the contents of *self* into *ary* or a newly allocated
        :mod:`numpy.ndarray`. If *ary* is given, it must have the right
        size (not necessarily shape) and dtype. If it is not given,
        a *pagelocked* specifies whether the new array is allocated
        page-locked.

    .. method :: get_async(ary=None, stream=None)

        Transfer the contents of *self* into *ary* or a newly allocated
        :mod:`numpy.ndarray`. If *ary* is given, it must have the right
        size (not necessarily shape) and dtype. If it is not given,
        a page-locked* array is newly allocated.

    .. method :: mul_add(self, selffac, other, otherfac, add_timer=None, stream=None):

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

    .. method :: fill(scalar, stream=None)

        Fill the array with *scalar*.

    .. method :: astype(dtype, stream=None)

        Return *self*, cast to *dtype*.

    .. attribute :: real

        Return the real part of *self*, or *self* if it is real.

        .. versionadded:: 0.94

    .. attribute :: imag

        Return the imaginary part of *self*, or *zeros_like(self)* if it is real.

        .. versionadded: 0.94

    .. method :: conj()

        Return the complex conjugate of *self*, or *self* if it is real.

        .. versionadded: 0.94

    .. method:: bind_to_texref(texref, allow_offset=False)

        Bind *self* to the :class:`pycuda.driver.TextureReference` *texref*.

        Due to alignment requirements, the effective texture bind address may be
        different from the requested one by an offset. This method returns this
        offset in units of *self*'s data type.  If *allow_offset* is ``False``, a
        nonzero value of this offset will cause an exception to be raised.

        .. note::

            It is recommended to use :meth:`bind_to_texref_ext` instead of
            this method.

    .. method:: bind_to_texref_ext(texref, channels=1, allow_double_hack=False, allow_offset=False)

        Bind *self* to the :class:`pycuda.driver.TextureReference` *texref*.
        In addition, set the texture reference's format to match :attr:`dtype`
        and its channel count to *channels*. This routine also sets the
        texture reference's :data:`pycuda.driver.TRSF_READ_AS_INTEGER` flag,
        if necessary.

        Due to alignment requirements, the effective texture bind address may be
        different from the requested one by an offset. This method returns this
        offset in units of *self*'s data type.  If *allow_offset* is ``False``, a
        nonzero value of this offset will cause an exception to be raised.

        .. versionadded:: 0.93

        .. highlight:: c

        As of this writing, CUDA textures do not natively support double-precision 
        floating point data. To remedy this deficiency, PyCUDA contains a workaround,
        which can be enabled by passing *True* for allow_double_hack. In this case,
        use the following code for texture access in your kernel code::

            #include <pycuda-helpers.hpp>

            texture<fp_tex_double, 1, cudaReadModeElementType> my_tex;

            __global__ void f()
            {
              ...
              fp_tex1Dfetch(my_tex, threadIdx.x);
              ...
            }

        .. highlight:: python

        (This workaround was added in version 0.94.)

Constructing :class:`GPUArray` Instances
----------------------------------------

.. function:: to_gpu(ary, allocator=None)

    Return a :class:`GPUArray` that is an exact copy of the :class:`numpy.ndarray`
    instance *ary*.

    See :class:`GPUArray` for the meaning of *allocator*.

.. function:: to_gpu_async(ary, allocator=None, stream=None)

    Return a :class:`GPUArray` that is an exact copy of the :class:`numpy.ndarray`
    instance *ary*. The copy is done asynchronously, optionally sequenced into
    *stream*.

    See :class:`GPUArray` for the meaning of *allocator*.

.. function:: empty(shape, dtype)

    A synonym for the :class:`GPUArray` constructor.

.. function:: zeros(shape, dtype)

    Same as :func:`empty`, but the :class:`GPUArray` is zero-initialized before
    being returned.

.. function:: empty_like(other_ary)

    Make a new, uninitialized :class:`GPUArray` having the same properties
    as *other_ary*.

.. function:: zeros_like(other_ary)

    Make a new, zero-initialized :class:`GPUArray` having the same properties
    as *other_ary*.

.. function:: arange(start, stop, step, dtype=None, stream=None)

    Create a :class:`GPUArray` filled with numbers spaced `step` apart,
    starting from `start` and ending at `stop`.

    For floating point arguments, the length of the result is
    `ceil((stop - start)/step)`.  This rule may result in the last
    element of the result being greater than `stop`.

    *dtype*, if not specified, is taken as the largest common type
    of *start*, *stop* and *step*.

.. function:: take(a, indices, stream=None)

    Return the :class:`GPUArray` ``[a[indices[0]], ..., a[indices[n]]]``.
    For the moment, *a* must be a type that can be bound to a texture.

Conditionals
^^^^^^^^^^^^

.. function:: if_positive(criterion, then_, else_, out=None, stream=None)

    Return an array like *then_*, which, for the element at index *i*,
    contains *then_[i]* if *criterion[i]>0*, else *else_[i]*. (added in 0.94)

.. function:: maximum(a, b, out=None, stream=None)

    Return the elementwise maximum of *a* and *b*. (added in 0.94)

.. function:: minimum(a, b, out=None, stream=None)

    Return the elementwise minimum of *a* and *b*. (added in 0.94)

Reductions
^^^^^^^^^^

.. function:: sum(a, dtype=None, stream=None)

.. function:: dot(a, b, dtype=None, stream=None)

.. function:: subset_dot(subset, a, b, dtype=None, stream=None)

.. function:: max(a, stream=None)

.. function:: min(a, stream=None)

.. function:: subset_max(subset, a, stream=None)

.. function:: subset_min(subset, a, stream=None)

Elementwise Functions on :class:`GPUArrray` Instances
-----------------------------------------------------

.. module:: pycuda.cumath

The :mod:`pycuda.cumath` module contains elementwise
workalikes for the functions contained in :mod:`math`.

Rounding and Absolute Value
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: fabs(array, stream=None)
.. function:: ceil(array, stream=None)
.. function:: floor(array, stream=None)

Exponentials, Logarithms and Roots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: exp(array, stream=None)
.. function:: log(array, stream=None)
.. function:: log10(array, stream=None)
.. function:: sqrt(array, stream=None)

Trigonometric Functions
^^^^^^^^^^^^^^^^^^^^^^^

.. function:: sin(array, stream=None)
.. function:: cos(array, stream=None)
.. function:: tan(array, stream=None)
.. function:: asin(array, stream=None)
.. function:: acos(array, stream=None)
.. function:: atan(array, stream=None)

Hyperbolic Functions
^^^^^^^^^^^^^^^^^^^^

.. function:: sinh(array, stream=None)
.. function:: cosh(array, stream=None)
.. function:: tanh(array, stream=None)

Floating Point Decomposition and Assembly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: fmod(arg, mod, stream=None)

    Return the floating point remainder of the division `arg/mod`,
    for each element in `arg` and `mod`.

.. function:: frexp(arg, stream=None)

    Return a tuple `(significands, exponents)` such that
    `arg == significand * 2**exponent`.

.. function:: ldexp(significand, exponent, stream=None)

    Return a new array of floating point values composed from the
    entries of `significand` and `exponent`, paired together as
    `result = significand * 2**exponent`.

.. function:: modf(arg, stream=None)

    Return a tuple `(fracpart, intpart)` of arrays containing the
    integer and fractional parts of `arg`.

Generating Arrays of Random Numbers
-----------------------------------

.. module:: pycuda.curandom

.. function:: rand(shape, dtype=numpy.float32, stream=None)

    Return an array of `shape` filled with random values of `dtype`
    in the range [0,1).

Single-pass Custom Expression Evaluation
----------------------------------------

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
    and *options* are passed unmodified to :class:`pycuda.compiler.SourceModule`.

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

Custom Reductions
-----------------

.. module:: pycuda.reduction

.. class:: ReductionKernel(dtype_out, neutral, reduce_expr, map_expr=None, arguments=None, name="reduce_kernel", keep=False, options=[], preamble="")

    Generate a kernel that takes a number of scalar or vector *arguments*
    (at least one vector argument), performs the *map_expr* on each entry of
    the vector argument and then the *reduce_expr* on the outcome of that.
    *neutral* serves as an initial value. *preamble* offers the possibility
    to add preprocessor directives and other code (such as helper functions)
    to be added before the actual reduction kernel code.

    Vectors in *map_expr* should be indexed by the variable *i*. *reduce_expr*
    uses the formal values "a" and "b" to indicate two operands of a binary
    reduction operation. If you do not specify a *map_expr*, "in[i]" -- and
    therefore the presence of only one input argument -- is automatically
    assumed.

    *dtype_out* specifies the :class:`numpy.dtype` in which the reduction is
    performed and in which the result is returned. *neutral* is
    specified as float or integer formatted as string. *reduce_expr* and
    *map_expr* are specified as string formatted operations and *arguments*
    is specified as a string formatted as a C argument list. *name* specifies
    the name as which the kernel is compiled, *keep* and *options* are passed
    unmodified to :class:`pycuda.compiler.SourceModule`. *preamble* is specified
    as a string of code.

    .. method __call__(*args, stream=None)

Here's a usage example::

    a = gpuarray.arange(400, dtype=numpy.float32)
    b = gpuarray.arange(400, dtype=numpy.float32)

    krnl = ReductionKernel(numpy.float32, neutral="0",
            reduce_expr="a+b", map_expr="a[i]*b[i]",
            arguments="float *a, float *b")

    my_dot_prod = krnl(a, b).get()

Fast Fourier Transforms
-----------------------

Bogdan Opanchuk's `pycudafft <http://pypi.python.org/pypi/pycudafft>`_ offers a
variety of GPU-based FFT implementations designed to work with
:class:`pycuda.gpuarray.GPUArray` objects.
