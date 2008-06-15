Tutorial Introduction
=====================

Getting started
---------------

Before you can use PyCuda, you have to initialize it and create a 
context::

  import pycuda.driver as cuda

  cuda.init()
  assert cuda.Device.count() >= 1

  dev = cuda.Device(0)
  ctx = dev.make_context()

Transferring Data
-----------------

The next step in most programs is to transfer data onto the device.
In PyCuda, you will mostly transfer data from :mod:`numpy` arrays
on the host. (But indeed, everything that satifies the Python buffer
interface will work, even a :class:`str`.) Let's make a 4x4 array 
of random numbers::

  import numpy
  a = numpy.random.randn(4,4)

But wait--*a* consists of double precision numbers, but Nvidia 
devices only support single precision as of this writing::

  a = a.astype(numpy.float32)

Finally, we need somewhere to transfer data to, so we need to 
allocate memory on the device::

  a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)

As a last step, we transfer to the data to the GPU::

  cuda.memcpy_htod(a_gpu, a)

Executing a Kernel
------------------

For this tutorial, we'll stick to something simple: We will write code to
double each entry in *a_gpu*. To this end, we write the corresponding CUDA C
code, and feed it into the constructor of a
:class:`pycuda.driver.SourceModule`::

  mod = cuda.SourceModule("""
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

(You can find the code for this demo as :file:`test/demo.py` in the PyCuda
source distribution.)

Bonus: Abstracting Away the Complications
-----------------------------------------
  
Using a :class:`pycuda.gpuarray.GPUArray`, the same effect can be 
achieved with much less writing::

  import pycuda.gpuarray as gpuarray
  a_gpu = gpuarray.to_gpu(numpy.random.randn(4,4).astype(numpy.float32))
  a_doubled = (2*a_gpu).get()
  print a_doubled
  print a_gpu

Where to Go from Here
---------------------

Once you feel sufficiently familiar with the basics, feel free to dig into the
:ref:`reference-doc`. Also check out PyCuda's test suite at
:file:`test/test_driver.py`. It contains examples (and tests!) of many more
advanced techniques.
