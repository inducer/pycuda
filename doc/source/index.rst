Welcome to PyCuda's documentation!
==================================

PyCuda gives you easy, Pythonic access to `Nvidia
<http://nvidia.com>`_'s `CUDA <http://nvidia.com/cuda/>`_ parallel
computation API. It exposes all low-level abstractions of the Nvidia
API, including

* Asynchronous execution (Streams, Events)
* Multi-D synchronous and asynchronous memory transfers
* Textures and Arrays

In addition, it has many convenience options that make programming
simple things simple. Here is a straightforward PyCuda program::

  import pycuda.driver as drv
  import numpy

  drv.init()
  dev = drv.Device(0)
  ctx = dev.make_context()

  mod = drv.SourceModule("""
  __global__ void multiply_them(float *dest, float *a, float *b)
  {
    const int i = threadIdx.x;
    dest[i] = a[i] * b[i];
  }
  """)

  multiply_them = mod.get_function("multiply_them")

  a = numpy.random.randn(400).astype(numpy.float32)
  b = numpy.random.randn(400).astype(numpy.float32)

  dest = numpy.zeros_like(a)
  multiply_them(
          drv.Out(dest), drv.In(a), drv.In(b),
          block=(400,1,1))

  print dest-a*b

On the surface, this program will print a screenful of zeros. Behind
the scenes, a lot more interesting stuff is going on:

* PyCuda has compiled the CUDA source code and uploaded it to the card. 
  
  .. note:: This code doesn't have to be a constant--you can easily have Python
    generate the code you want to compile.

* PyCuda's numpy interaction code has automatically allocated
  space on the device, copied the numpy arrays *a* and *b* over,
  launched a 400x1x1 single-block grid, and copied *dest* back.

  Note that you can just as well keep your data on the card between
  kernel invocations--no need to copy data all the time.

* PyCuda automatically frees the resources you use, such as
  memory, modules, contexts, taking into account any ordering 
  constraints.

Curious? Let's get started.

Contents
=========

.. toctree::
    :maxdepth: 2

    install
    tutorial
    driver
    array

Note that this guide will not explain CUDA programming and technology.  Please
refer to Nvidia's `programming documentation
<http://www.nvidia.com/object/cuda_learn.html>`_ for that.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

