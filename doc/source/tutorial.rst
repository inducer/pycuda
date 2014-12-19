Tutorial
========

Getting started
---------------

Before you can use PyCuda, you have to import and initialize it::

  import pycuda.driver as cuda
  import pycuda.autoinit
  from pycuda.compiler import SourceModule

Note that you do not *have* to use :mod:`pycuda.autoinit`--
initialization, context creation, and cleanup can also be performed
manually, if desired.

Transferring Data
-----------------

The next step in most programs is to transfer data onto the device.
In PyCuda, you will mostly transfer data from :mod:`numpy` arrays
on the host. (But indeed, everything that satisfies the Python buffer
interface will work, even a :class:`str`.) Let's make a 4x4 array
of random numbers::

  import numpy
  a = numpy.random.randn(4,4)

But wait--*a* consists of double precision numbers, but most nVidia
devices only support single precision::

  a = a.astype(numpy.float32)

Finally, we need somewhere to transfer data to, so we need to
allocate memory on the device::

  a_gpu = cuda.mem_alloc(a.nbytes)

As a last step, we need to transfer the data to the GPU::

  cuda.memcpy_htod(a_gpu, a)

Executing a Kernel
------------------

For this tutorial, we'll stick to something simple: We will write code to
double each entry in *a_gpu*. To this end, we write the corresponding CUDA C
code, and feed it into the constructor of a
:class:`pycuda.compiler.SourceModule`::

  mod = SourceModule("""
    __global__ void doublify(float *a)
    {
      int idx = threadIdx.x + threadIdx.y*4;
      a[idx] *= 2;
    }
    """)

If there aren't any errors, the code is now compiled and loaded onto the
device. We find a reference to our :class:`pycuda.driver.Function` and call
it, specifying *a_gpu* as the argument, and a block size of 4x4::

  func = mod.get_function("doublify")
  func(a_gpu, block=(4,4,1))

Finally, we fetch the data back from the GPU and display it, together with the
original *a*::

  a_doubled = numpy.empty_like(a)
  cuda.memcpy_dtoh(a_doubled, a_gpu)
  print a_doubled
  print a

This will print something like this::

  [[ 0.51360393  1.40589952  2.25009012  3.02563429]
   [-0.75841576 -1.18757617  2.72269917  3.12156057]
   [ 0.28826082 -2.92448163  1.21624792  2.86353827]
   [ 1.57651746  0.63500965  2.21570683 -0.44537592]]
  [[ 0.25680196  0.70294976  1.12504506  1.51281714]
   [-0.37920788 -0.59378809  1.36134958  1.56078029]
   [ 0.14413041 -1.46224082  0.60812396  1.43176913]
   [ 0.78825873  0.31750482  1.10785341 -0.22268796]]

It worked! That completes our walkthrough. Thankfully, PyCuda takes
over from here and does all the cleanup for you, so you're done.
Stick around for some bonus material in the next section, though.

(You can find the code for this demo as :file:`examples/demo.py` in the PyCuda
source distribution.)

Shortcuts for Explicit Memory Copies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`pycuda.driver.In`, :class:`pycuda.driver.Out`, and
:class:`pycuda.driver.InOut` argument handlers can simplify some of the memory
transfers. For example, instead of creating *a_gpu*, if replacing *a* is fine,
the following code can be used::

  func(cuda.InOut(a), block=(4, 4, 1))

Prepared Invocations
^^^^^^^^^^^^^^^^^^^^

Function invocation using the built-in :meth:`pycuda.driver.Function.__call__`
method incurs overhead for type identification (see :ref:`reference-doc`). To
achieve the same effect as above without this overhead, the function is bound
to argument types (as designated by Python's standard library :mod:`struct`
module), and then called. This also avoids having to assign explicit argument
sizes using the `numpy.number` classes::

    grid = (1, 1)
    block = (4, 4, 1)
    func.prepare("P")
    func.prepared_call(grid, block, a_gpu)

Bonus: Abstracting Away the Complications
-----------------------------------------

Using a :class:`pycuda.gpuarray.GPUArray`, the same effect can be
achieved with much less writing::

  import pycuda.gpuarray as gpuarray
  import pycuda.driver as cuda
  import pycuda.autoinit
  import numpy

  a_gpu = gpuarray.to_gpu(numpy.random.randn(4,4).astype(numpy.float32))
  a_doubled = (2*a_gpu).get()
  print a_doubled
  print a_gpu

Advanced Topics
---------------

Structures
^^^^^^^^^^

(contributed by Nicholas Tung, find the code in :file:`examples/demo_struct.py`)

Suppose we have the following structure, for doubling a number of variable
length arrays::

  mod = SourceModule("""
      struct DoubleOperation {
          int datalen, __padding; // so 64-bit ptrs can be aligned
          float *ptr;
      };

      __global__ void double_array(DoubleOperation *a) {
          a = &a[blockIdx.x];
          for (int idx = threadIdx.x; idx < a->datalen; idx += blockDim.x) {
              a->ptr[idx] *= 2;
          }
      }
      """)

Each block in the grid (see CUDA documentation) will double one of the arrays.
The `for` loop allows for more data elements than threads to be doubled,
though is not efficient if one can guarantee that there will be a sufficient
number of threads. Next, a wrapper class for the structure is created, and
two arrays are instantiated::

  class DoubleOpStruct:
      mem_size = 8 + numpy.intp(0).nbytes
      def __init__(self, array, struct_arr_ptr):
          self.data = cuda.to_device(array)
          self.shape, self.dtype = array.shape, array.dtype
          cuda.memcpy_htod(int(struct_arr_ptr), numpy.getbuffer(numpy.int32(array.size)))
          cuda.memcpy_htod(int(struct_arr_ptr) + 8, numpy.getbuffer(numpy.intp(int(self.data))))
      def __str__(self):
          return str(cuda.from_device(self.data, self.shape, self.dtype))

  struct_arr = cuda.mem_alloc(2 * DoubleOpStruct.mem_size)
  do2_ptr = int(struct_arr) + DoubleOpStruct.mem_size

  array1 = DoubleOpStruct(numpy.array([1, 2, 3], dtype=numpy.float32), struct_arr)
  array2 = DoubleOpStruct(numpy.array([0, 4], dtype=numpy.float32), do2_ptr)
  print("original arrays", array1, array2)

This code uses the :func:`pycuda.driver.to_device` and
:func:`pycuda.driver.from_device` functions to allocate and copy values, and
demonstrates how offsets to an allocated block of memory can be used. Finally,
the code can be executed; the following demonstrates doubling both arrays, then
only the second::

  func = mod.get_function("double_array")
  func(struct_arr, block = (32, 1, 1), grid=(2, 1))
  print("doubled arrays", array1, array2)

  func(numpy.intp(do2_ptr), block = (32, 1, 1), grid=(1, 1))
  print("doubled second only", array1, array2, "\n")

Where to go from here
---------------------

Once you feel sufficiently familiar with the basics, feel free to dig into the
:ref:`reference-doc`. For more examples, check the in the :file:`examples/`
subdirectory of the distribution.  This folder also contains several benchmarks
to see the difference between GPU and CPU based calculations. As a reference for
how stuff is done, PyCuda's test suite in the :file:`test/` subdirectory of the
distribution may also be of help.
