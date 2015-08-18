GPU Arrays
==========

.. module:: pycuda.gpuarray

Vector Types
------------

.. class :: vec

    All of CUDA's supported vector types, such as `float3` and `long4` are
    available as :mod:`numpy` data types within this class. These
    :class:`numpy.dtype` instances have field names of `x`, `y`, `z`, and `w`
    just like their CUDA counterparts. They will work both for parameter passing
    to kernels as well as for passing data back and forth between kernels and
    Python code. For each type, a `make_type` function is also provided (e.g.
    `make_float3(x,y,z)`).

The :class:`GPUArray` Array Class
---------------------------------

.. class:: GPUArray(shape, dtype, *, allocator=None, order="C")

    A :class:`numpy.ndarray` work-alike that stores its data and performs its
    computations on the compute device.  *shape* and *dtype* work exactly as in
    :mod:`numpy`.  Arithmetic methods in :class:`GPUArray` support the
    broadcasting of scalars. (e.g. `array+5`) If the

    *allocator* is a callable that, upon being called with an argument of the number
    of bytes to be allocated, returns an object that can be cast to an
    :class:`int` representing the address of the newly allocated memory.
    Observe that both :func:`pycuda.driver.mem_alloc` and
    :meth:`pycuda.tools.DeviceMemoryPool.alloc` are a model of this interface.

    All arguments beyond *allocator* should be considered keyword-only.

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

    .. attribute :: strides

        Tuple of bytes to step in each dimension when traversing an array.

    .. attribute :: flags

        Return an object with attributes `c_contiguous`, `f_contiguous` and `forc`,
        which may be used to query contiguity properties in analogy to
        :attr:`numpy.ndarray.flags`.

    .. attribute :: ptr

        Return an :class:`int` reflecting the address in device memory where
        this array resides.

        .. versionadded: 2011.1

    .. method :: __len__()

        Returns the size of the leading dimension of *self*.

      .. warning ::

        This method existed in version 0.93 and below, but it returned the value
        of :attr:`size` instead of its current value. The change was made in order
        to match :mod:`numpy`.

    .. method :: reshape(shape)

        Returns an array containing the same data with a new shape.

    .. method :: ravel()

        Returns flattened array containing the same data.

    .. method :: view(dtype=None)

        Returns view of array with the same data. If *dtype* is different from
        current dtype, the actual bytes of memory will be reinterpreted.

    .. method :: squeeze(dtype=None)

        Returns a view of the array with dimensions of length 1 removed.

        .. versionadded: 2015.1.4

    .. method :: set(ary)

        Transfer the contents the :class:`numpy.ndarray` object *ary*
        onto the device.

        *ary* must have the same dtype and size (not necessarily shape) as *self*.

    .. method :: set_async(ary, stream=None)

        Asynchronously transfer the contents the :class:`numpy.ndarray` object *ary*
        onto the device, optionally sequenced on *stream*.

        *ary* must have the same dtype and size (not necessarily shape) as *self*.

    .. method :: get(ary=None, pagelocked=False)

        Transfer the contents of *self* into *ary* or a newly allocated
        :mod:`numpy.ndarray`. If *ary* is given, it must have the same
        shape and dtype. If it is not given,
        a *pagelocked* specifies whether the new array is allocated
        page-locked.

        .. versionchanged:: 2015.2

            *ary* with different shape was deprecated.

    .. method :: get_async(stream=None, ary=None)

        Transfer the contents of *self* into *ary* or a newly allocated
        :mod:`numpy.ndarray`. If *ary* is given, it must have the right
        size (not necessarily shape) and dtype. If it is not given,
        a *page-locked* array is newly allocated.

    .. method :: copy()

        .. versionadded :: 2013.1

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

.. function:: empty(shape, dtype, *, allocator=None, order="C")

    A synonym for the :class:`GPUArray` constructor.

.. function:: zeros(shape, dtype, *, allocator=None, order="C")

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

.. function:: subset_sum(subset, a, dtype=None, stream=None)

    .. versionadded:: 2013.1

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

.. function:: fabs(array, *, out=None, stream=None)
.. function:: ceil(array, *, out=None, stream=None)
.. function:: floor(array, *, out=None, stream=None)

Exponentials, Logarithms and Roots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: exp(array, *, out=None, stream=None)
.. function:: log(array, *, out=None, stream=None)
.. function:: log10(array, *, out=None, stream=None)
.. function:: sqrt(array, *, out=None, stream=None)

Trigonometric Functions
^^^^^^^^^^^^^^^^^^^^^^^

.. function:: sin(array, *, out=None, stream=None)
.. function:: cos(array, *, out=None, stream=None)
.. function:: tan(array, *, out=None, stream=None)
.. function:: asin(array, *, out=None, stream=None)
.. function:: acos(array, *, out=None, stream=None)
.. function:: atan(array, *, out=None, stream=None)

Hyperbolic Functions
^^^^^^^^^^^^^^^^^^^^

.. function:: sinh(array, *, out=None, stream=None)
.. function:: cosh(array, *, out=None, stream=None)
.. function:: tanh(array, *, out=None, stream=None)

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

    .. note::

        The use case for this function is "I need some random numbers.
        I don't care how good they are or how fast I get them." It uses
        a pretty terrible MD5-based generator and doesn't even attempt
        to cache generated code.

        If you're interested in a non-toy random number generator, use the
        CURAND-based functionality below.

.. warning::

    The following classes are using random number generators that run on the GPU.
    Each thread uses its own generator. Creation of those generators requires more
    resources than subsequent generation of random numbers. After experiments
    it looks like maximum number of active generators on Tesla devices
    (with compute capabilities 1.x) is 256. Fermi devices allow for creating
    1024 generators without any problems. If there are troubles with creating
    objects of class PseudoRandomNumberGenerator or QuasiRandomNumberGenerator
    decrease number of created generators
    (and therefore number of active threads).

A pseudorandom sequence of numbers satisfies most of the statistical properties
of a truly random sequence but is generated by a deterministic algorithm.  A
quasirandom sequence of n-dimensional points is generated by a deterministic
algorithm designed to fill an n-dimensional space evenly.

Quasirandom numbers are more expensive to generate.

.. function:: get_curand_version()

    Obtain the version of CURAND against which PyCUDA was compiled. Returns a
    3-tuple of integers as *(major, minor, revision)*.

.. function:: seed_getter_uniform(N)

    Return an :class:`GPUArray` filled with one random `int32` repeated `N`
    times which can be used as a seed for XORWOW generator.

.. function:: seed_getter_unique(N)

    Return an :class:`GPUArray` filled with `N` random `int32` which can
    be used as a seed for XORWOW generator.

.. class:: XORWOWRandomNumberGenerator(seed_getter=None, offset=0)

    :arg seed_getter: a function that, given an integer count, will yield an
      `int32` :class:`GPUArray` of seeds.
    :arg offset: Starting index into the XORWOW sequence, given seed.

    Provides pseudorandom numbers. Generates sequences with period
    at least :math:`2^190`.

    CUDA 3.2 and above.

    .. versionadded:: 2011.1

    .. method:: fill_uniform(data, stream=None)

        Fills in :class:`GPUArray` *data* with uniformly distributed
        pseudorandom values.

    .. method:: gen_uniform(shape, dtype, stream=None)

        Creates object of :class:`GPUArray` with given *shape* and *dtype*,
        fills it in with uniformly distributed pseudorandom values,
        and returns newly created object.

    .. method:: fill_normal(data, stream=None)

        Fills in :class:`GPUArray` *data* with normally distributed
        pseudorandom values.

    .. method:: gen_normal(shape, dtype, stream=None)

        Creates object of :class:`GPUArray` with given *shape* and *dtype*,
        fills it in with normally distributed pseudorandom values,
        and returns newly created object.

    .. method:: fill_log_normal(data, mean, stddev, stream=None)

        Fills in :class:`GPUArray` *data* with log-normally distributed
        pseudorandom values with mean *mean* and standard deviation *stddev*.

        CUDA 4.0 and above.

        .. versionadded:: 2012.2

    .. method:: gen_log_normal(shape, dtype, mean, stddev, stream=None)

        Creates object of :class:`GPUArray` with given *shape* and *dtype*,
        fills it in with log-normally distributed pseudorandom values
        with mean *mean* and standard deviation *stddev*, and returns
        newly created object.

        CUDA 4.0 and above.

        .. versionadded:: 2012.2

    .. method:: fill_poisson(data, lambda_value, stream=None)

        Fills in :class:`GPUArray` *data* with Poisson distributed
        pseudorandom values with lambda *lambda_value*. *data* must
        be of type 32-bit unsigned int.

        CUDA 5.0 and above.

        .. versionadded:: 2013.1

    .. method:: gen_poisson(shape, dtype, lambda_value, stream=None)

        Creates object of :class:`GPUArray` with given *shape* and *dtype*,
        fills it in with Poisson distributed pseudorandom values
        with lambda *lambda_value*, and returns newly created object.
        *dtype* must be 32-bit unsigned int.

        CUDA 5.0 and above.

        .. versionadded:: 2013.1

    .. method:: call_skip_ahead(i, stream=None)

        Forces all generators to skip i values. Is equivalent to generating
        i values and discarding results, but is much faster.

    .. method::  call_skip_ahead_array(i, stream=None)

        Accepts array i of integer values, telling each generator how many
        values to skip.

    .. method:: call_skip_ahead_sequence(i, stream=None)

        Forces all generators to skip i subsequences. Is equivalent to
        generating i * :math:`2^67` values and discarding results,
        but is much faster.

    .. method:: call_skip_ahead_sequence_array(i, stream=None)

        Accepts array i of integer values, telling each generator how many
        subsequences to skip.

.. class:: MRG32k3aRandomNumberGenerator(seed_getter=None, offset=0)

    :arg seed_getter: a function that, given an integer count, will yield an
      `int32` :class:`GPUArray` of seeds.
    :arg offset: Starting index into the XORWOW sequence, given seed.

    Provides pseudorandom numbers. Generates sequences with period
    at least :math:`2^190`.

    CUDA 4.1 and above.

    .. versionadded:: 2013.1

    .. method:: fill_uniform(data, stream=None)

        Fills in :class:`GPUArray` *data* with uniformly distributed
        pseudorandom values.

    .. method:: gen_uniform(shape, dtype, stream=None)

        Creates object of :class:`GPUArray` with given *shape* and *dtype*,
        fills it in with uniformly distributed pseudorandom values,
        and returns newly created object.

    .. method:: fill_normal(data, stream=None)

        Fills in :class:`GPUArray` *data* with normally distributed
        pseudorandom values.

    .. method:: gen_normal(shape, dtype, stream=None)

        Creates object of :class:`GPUArray` with given *shape* and *dtype*,
        fills it in with normally distributed pseudorandom values,
        and returns newly created object.

    .. method:: fill_log_normal(data, mean, stddev, stream=None)

        Fills in :class:`GPUArray` *data* with log-normally distributed
        pseudorandom values with mean *mean* and standard deviation *stddev*.

    .. method:: gen_log_normal(shape, dtype, mean, stddev, stream=None)

        Creates object of :class:`GPUArray` with given *shape* and *dtype*,
        fills it in with log-normally distributed pseudorandom values
        with mean *mean* and standard deviation *stddev*, and returns
        newly created object.

    .. method:: fill_poisson(data, lambda_value, stream=None)

        Fills in :class:`GPUArray` *data* with Poisson distributed
        pseudorandom values with lambda *lambda_value*. *data* must
        be of type 32-bit unsigned int.

        CUDA 5.0 and above.

        .. versionadded:: 2013.1

    .. method:: gen_poisson(shape, dtype, lambda_value, stream=None)

        Creates object of :class:`GPUArray` with given *shape* and *dtype*,
        fills it in with Poisson distributed pseudorandom values
        with lambda *lambda_value*, and returns newly created object.
        *dtype* must be 32-bit unsigned int.

        CUDA 5.0 and above.

        .. versionadded:: 2013.1

    .. method:: call_skip_ahead(i, stream=None)

        Forces all generators to skip i values. Is equivalent to generating
        i values and discarding results, but is much faster.

    .. method::  call_skip_ahead_array(i, stream=None)

        Accepts array i of integer values, telling each generator how many
        values to skip.

    .. method:: call_skip_ahead_sequence(i, stream=None)

        Forces all generators to skip i subsequences. Is equivalent to
        generating i * :math:`2^67` values and discarding results,
        but is much faster.

    .. method:: call_skip_ahead_sequence_array(i, stream=None)

        Accepts array i of integer values, telling each generator how many
        subsequences to skip.

.. function:: generate_direction_vectors(count, direction=direction_vector_set.VECTOR_32)

    Return an :class:`GPUArray` `count` filled with direction vectors
    used to initialize Sobol generators.

.. function:: generate_scramble_constants32(count)

    Return a :class:`GPUArray` filled with `count' 32-bit unsigned integer
    numbers used to initialize :class:`ScrambledSobol32RandomNumberGenerator`

.. function:: generate_scramble_constants64(count)

    Return a :class:`GPUArray` filled with `count' 64-bit unsigned integer
    numbers used to initialize :class:`ScrambledSobol64RandomNumberGenerator`

.. class:: Sobol32RandomNumberGenerator(dir_vector=None, offset=0)

    :arg dir_vector: a :class:`GPUArray` of 32-element `int32` vectors which
      are used to initialize quasirandom generator; it must contain one vector
      for each initialized generator
    :arg offset: Starting index into the Sobol32 sequence, given direction
      vector.

    Provides quasirandom numbers. Generates
    sequences with period of :math:`2^32`.

    CUDA 3.2 and above.

    .. versionadded:: 2011.1

    .. method:: fill_uniform(data, stream=None)

        Fills in :class:`GPUArray` *data* with uniformly distributed
        quasirandom values.

    .. method:: gen_uniform(shape, dtype, stream=None)

        Creates object of :class:`GPUArray` with given *shape* and *dtype*,
        fills it in with uniformly distributed pseudorandom values,
        and returns newly created object.

    .. method:: fill_normal(data, stream=None)

        Fills in :class:`GPUArray` *data* with normally distributed
        quasirandom values.

    .. method:: gen_normal(shape, dtype, stream=None)

        Creates object of :class:`GPUArray` with given *shape* and *dtype*,
        fills it in with normally distributed pseudorandom values,
        and returns newly created object.

    .. method:: fill_log_normal(data, mean, stddev, stream=None)

        Fills in :class:`GPUArray` *data* with log-normally distributed
        pseudorandom values with mean *mean* and standard deviation *stddev*.

        CUDA 4.0 and above.

        .. versionadded:: 2012.2

    .. method:: gen_log_normal(shape, dtype, mean, stddev, stream=None)

        Creates object of :class:`GPUArray` with given *shape* and *dtype*,
        fills it in with log-normally distributed pseudorandom values
        with mean *mean* and standard deviation *stddev*, and returns
        newly created object.

        CUDA 4.0 and above.

        .. versionadded:: 2012.2

    .. method:: fill_poisson(data, lambda_value, stream=None)

        Fills in :class:`GPUArray` *data* with Poisson distributed
        pseudorandom values with lambda *lambda_value*. *data* must
        be of type 32-bit unsigned int.

        CUDA 5.0 and above.

        .. versionadded:: 2013.1

    .. method:: gen_poisson(shape, dtype, lambda_value, stream=None)

        Creates object of :class:`GPUArray` with given *shape* and *dtype*,
        fills it in with Poisson distributed pseudorandom values
        with lambda *lambda_value*, and returns newly created object.
        *dtype* must be 32-bit unsigned int.

        CUDA 5.0 and above.

        .. versionadded:: 2013.1

    .. method:: call_skip_ahead(i, stream=None)

        Forces all generators to skip i values. Is equivalent to generating
        i values and discarding results, but is much faster.

    .. method:: call_skip_ahead_array(i, stream=None)

        Accepts array i of integer values, telling each generator how many
        values to skip.

.. class:: ScrambledSobol32RandomNumberGenerator(dir_vector=None, scramble_vector=None, offset=0)

    :arg dir_vector: a :class:`GPUArray` of 32-element `uint32` vectors which
      are used to initialize quasirandom generator; it must contain one vector
      for each initialized generator
    :arg scramble_vector: a :class:`GPUArray` of `uint32` elements which
      are used to initialize quasirandom generator; it must contain one number
      for each initialized generator
    :arg offset: Starting index into the Sobol32 sequence, given direction
      vector.

    Provides quasirandom numbers. Generates
    sequences with period of :math:`2^32`.

    CUDA 4.0 and above.

    .. versionadded:: 2011.1

    .. method:: fill_uniform(data, stream=None)

        Fills in :class:`GPUArray` *data* with uniformly distributed
        quasirandom values.

    .. method:: gen_uniform(shape, dtype, stream=None)

        Creates object of :class:`GPUArray` with given *shape* and *dtype*,
        fills it in with uniformly distributed pseudorandom values,
        and returns newly created object.

    .. method:: fill_normal(data, stream=None)

        Fills in :class:`GPUArray` *data* with normally distributed
        quasirandom values.

    .. method:: gen_normal(shape, dtype, stream=None)

        Creates object of :class:`GPUArray` with given *shape* and *dtype*,
        fills it in with normally distributed pseudorandom values,
        and returns newly created object.

    .. method:: fill_log_normal(data, mean, stddev, stream=None)

        Fills in :class:`GPUArray` *data* with log-normally distributed
        pseudorandom values with mean *mean* and standard deviation *stddev*.

        CUDA 4.0 and above.

        .. versionadded:: 2012.2

    .. method:: gen_log_normal(shape, dtype, mean, stddev, stream=None)

        Creates object of :class:`GPUArray` with given *shape* and *dtype*,
        fills it in with log-normally distributed pseudorandom values
        with mean *mean* and standard deviation *stddev*, and returns
        newly created object.

        CUDA 4.0 and above.

        .. versionadded:: 2012.2

    .. method:: fill_poisson(data, lambda_value, stream=None)

        Fills in :class:`GPUArray` *data* with Poisson distributed
        pseudorandom values with lambda *lambda_value*. *data* must
        be of type 32-bit unsigned int.

        CUDA 5.0 and above.

        .. versionadded:: 2013.1

    .. method:: gen_poisson(shape, dtype, lambda_value, stream=None)

        Creates object of :class:`GPUArray` with given *shape* and *dtype*,
        fills it in with Poisson distributed pseudorandom values
        with lambda *lambda_value*, and returns newly created object.
        *dtype* must be 32-bit unsigned int.

        CUDA 5.0 and above.

        .. versionadded:: 2013.1

    .. method:: call_skip_ahead(i, stream=None)

        Forces all generators to skip i values. Is equivalent to generating
        i values and discarding results, but is much faster.

    .. method:: call_skip_ahead_array(i, stream=None)

        Accepts array i of integer values, telling each generator how many
        values to skip.

.. class:: Sobol64RandomNumberGenerator(dir_vector=None, offset=0)

    :arg dir_vector: a :class:`GPUArray` of 64-element `uint64` vectors which
      are used to initialize quasirandom generator; it must contain one vector
      for each initialized generator
    :arg offset: Starting index into the Sobol64 sequence, given direction
      vector.

    Provides quasirandom numbers. Generates
    sequences with period of :math:`2^64`.

    CUDA 4.0 and above.

    .. versionadded:: 2011.1

    .. method:: fill_uniform(data, stream=None)

        Fills in :class:`GPUArray` *data* with uniformly distributed
        quasirandom values.

    .. method:: gen_uniform(shape, dtype, stream=None)

        Creates object of :class:`GPUArray` with given *shape* and *dtype*,
        fills it in with uniformly distributed pseudorandom values,
        and returns newly created object.

    .. method:: fill_normal(data, stream=None)

        Fills in :class:`GPUArray` *data* with normally distributed
        quasirandom values.

    .. method:: gen_normal(shape, dtype, stream=None)

        Creates object of :class:`GPUArray` with given *shape* and *dtype*,
        fills it in with normally distributed pseudorandom values,
        and returns newly created object.

    .. method:: fill_log_normal(data, mean, stddev, stream=None)

        Fills in :class:`GPUArray` *data* with log-normally distributed
        pseudorandom values with mean *mean* and standard deviation *stddev*.

        CUDA 4.0 and above.

        .. versionadded:: 2012.2

    .. method:: gen_log_normal(shape, dtype, mean, stddev, stream=None)

        Creates object of :class:`GPUArray` with given *shape* and *dtype*,
        fills it in with log-normally distributed pseudorandom values
        with mean *mean* and standard deviation *stddev*, and returns
        newly created object.

        CUDA 4.0 and above.

        .. versionadded:: 2012.2

    .. method:: fill_poisson(data, lambda_value, stream=None)

        Fills in :class:`GPUArray` *data* with Poisson distributed
        pseudorandom values with lambda *lambda_value*. *data* must
        be of type 32-bit unsigned int.

        CUDA 5.0 and above.

        .. versionadded:: 2013.1

    .. method:: gen_poisson(shape, dtype, lambda_value, stream=None)

        Creates object of :class:`GPUArray` with given *shape* and *dtype*,
        fills it in with Poisson distributed pseudorandom values
        with lambda *lambda_value*, and returns newly created object.
        *dtype* must be 32-bit unsigned int.

        CUDA 5.0 and above.

        .. versionadded:: 2013.1

    .. method:: call_skip_ahead(i, stream=None)

        Forces all generators to skip i values. Is equivalent to generating
        i values and discarding results, but is much faster.

    .. method:: call_skip_ahead_array(i, stream=None)

        Accepts array i of integer values, telling each generator how many
        values to skip.

.. class:: ScrambledSobol64RandomNumberGenerator(dir_vector=None, scramble_vector=None, offset=0)

    :arg dir_vector: a :class:`GPUArray` of 64-element `uint64` vectors which
      are used to initialize quasirandom generator; it must contain one vector
      for each initialized generator
    :arg scramble_vector: a :class:`GPUArray` of `uint64` vectors which
      are used to initialize quasirandom generator; it must contain one vector
      for each initialized generator
    :arg offset: Starting index into the ScrambledSobol64 sequence,
      given direction vector.

    Provides quasirandom numbers. Generates
    sequences with period of :math:`2^64`.

    CUDA 4.0 and above.

    .. versionadded:: 2011.1

    .. method:: fill_uniform(data, stream=None)

        Fills in :class:`GPUArray` *data* with uniformly distributed
        quasirandom values.

    .. method:: gen_uniform(shape, dtype, stream=None)

        Creates object of :class:`GPUArray` with given *shape* and *dtype*,
        fills it in with uniformly distributed pseudorandom values,
        and returns newly created object.

    .. method:: fill_normal(data, stream=None)

        Fills in :class:`GPUArray` *data* with normally distributed
        quasirandom values.

    .. method:: gen_normal(shape, dtype, stream=None)

        Creates object of :class:`GPUArray` with given *shape* and *dtype*,
        fills it in with normally distributed pseudorandom values,
        and returns newly created object.

    .. method:: fill_log_normal(data, mean, stddev, stream=None)

        Fills in :class:`GPUArray` *data* with log-normally distributed
        pseudorandom values with mean *mean* and standard deviation *stddev*.

        CUDA 4.0 and above.

        .. versionadded:: 2012.2

    .. method:: gen_log_normal(shape, dtype, mean, stddev, stream=None)

        Creates object of :class:`GPUArray` with given *shape* and *dtype*,
        fills it in with log-normally distributed pseudorandom values
        with mean *mean* and standard deviation *stddev*, and returns
        newly created object.

        CUDA 4.0 and above.

        .. versionadded:: 2012.2

    .. method:: fill_poisson(data, lambda_value, stream=None)

        Fills in :class:`GPUArray` *data* with Poisson distributed
        pseudorandom values with lambda *lambda_value*. *data* must
        be of type 32-bit unsigned int.

        CUDA 5.0 and above.

        .. versionadded:: 2013.1

    .. method:: gen_poisson(shape, dtype, lambda_value, stream=None)

        Creates object of :class:`GPUArray` with given *shape* and *dtype*,
        fills it in with Poisson distributed pseudorandom values
        with lambda *lambda_value*, and returns newly created object.
        *dtype* must be 32-bit unsigned int.

        CUDA 5.0 and above.

        .. versionadded:: 2013.1

    .. method:: call_skip_ahead(i, stream=None)

        Forces all generators to skip i values. Is equivalent to generating
        i values and discarding results, but is much faster.

    .. method:: call_skip_ahead_array(i, stream=None)

        Accepts array i of integer values, telling each generator how many
        values to skip.

Single-pass Custom Expression Evaluation
----------------------------------------

.. module:: pycuda.elementwise

Evaluating involved expressions on :class:`GPUArray` instances can be
somewhat inefficient, because a new temporary is created for each
intermediate result. The functionality in the module :mod:`pycuda.elementwise`
contains tools to help generate kernels that evaluate multi-stage expressions
on one or several operands in a single pass.

.. class:: ElementwiseKernel(arguments, operation, name="kernel", keep=False, options=[], preamble="")

    Generate a kernel that takes a number of scalar or vector *arguments*
    and performs the scalar *operation* on each entry of its arguments, if that
    argument is a vector.

    *arguments* is specified as a string formatted as a C argument list.
    *operation* is specified as a C assignment statement, without a semicolon.
    Vectors in *operation* should be indexed by the variable *i*.

    *name* specifies the name as which the kernel is compiled, *keep*
    and *options* are passed unmodified to :class:`pycuda.compiler.SourceModule`.

    *preamble* specifies some source code that is included before the
    elementwise kernel specification. You may use this to include other
    files and/or define functions that are used by *operation*.

    .. method:: __call__(*args, range=None, slice=None)

        Invoke the generated scalar kernel. The arguments may either be scalars or
        :class:`GPUArray` instances.

        If *range* is given, it must be a :class:`slice` object and specifies
        the range of indices *i* for which the *operation* is carried out.

        If *slice* is given, it must be a :class:`slice` object and specifies
        the range of indices *i* for which the *operation* is carried out,
        truncated to the container. Also, *slice* may contain negative indices
        to index relative to the end of the array.

        If *stream* is given, it must be a :class:`pycuda.driver.Stream` object,
        where the execution will be serialized.

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

.. class:: ReductionKernel(dtype_out, neutral, reduce_expr, map_expr=None, arguments=None, name="reduce_kernel", keep=False, options=[], preamble="", allocator=None)

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
            reduce_expr="a+b", map_expr="x[i]*y[i]",
            arguments="float *x, float *y")

    my_dot_prod = krnl(a, b).get()

Parallel Scan / Prefix Sum
--------------------------

.. module:: pycuda.scan

.. class:: ExclusiveScanKernel(dtype, scan_expr, neutral, name_prefix="scan", options=[], preamble="")

    Generates a kernel that can compute a `prefix sum <https://secure.wikimedia.org/wikipedia/en/wiki/Prefix_sum>`_
    using any associative operation given as *scan_expr*.
    *scan_expr* uses the formal values "a" and "b" to indicate two operands of
    an associative binary operation. *neutral* is the neutral element
    of *scan_expr*, obeying *scan_expr(a, neutral) == a*.

    *dtype* specifies the type of the arrays being operated on.
    *name_prefix* is used for kernel names to ensure recognizability
    in profiles and logs. *options* is a list of compiler options to use
    when building. *preamble* specifies a string of code that is
    inserted before the actual kernels.

    .. method:: __call__(self, input_ary, output_ary=None, allocator=None, queue=None)

.. class:: InclusiveScanKernel(dtype, scan_expr, neutral=None, name_prefix="scan", options=[], preamble="", devices=None)

    Works like :class:`ExclusiveScanKernel`. Unlike the exclusive case,
    *neutral* is not required.

Here's a usage example::

    knl = InclusiveScanKernel(np.int32, "a+b")

    n = 2**20-2**18+5
    host_data = np.random.randint(0, 10, n).astype(np.int32)
    dev_data = gpuarray.to_gpu(queue, host_data)

    knl(dev_data)
    assert (dev_data.get() == np.cumsum(host_data, axis=0)).all()

Custom data types in Reduction and Scan
---------------------------------------

If you would like to use your own (struct/union/whatever) data types in
scan and reduction, define those types in the *preamble* and let PyCUDA
know about them using this function:

.. function:: pycuda.tools.register_dtype(dtype, name)

    *dtype* is a :func:`numpy.dtype`.

    .. versionadded: 2011.2

GPGPU Algorithms
----------------

Bogdan Opanchuk's `reikna <http://pypi.python.org/pypi/reikna>`_ offers a
variety of GPU-based algorithms (FFT, RNG, matrix multiplication) designed to work with
:class:`pycuda.gpuarray.GPUArray` objects.
