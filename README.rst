PyCUDA: Pythonic Access to CUDA, with Arrays and Algorithms
=============================================================

.. image:: https://gitlab.tiker.net/inducer/pycuda/badges/main/pipeline.svg
    :alt: Gitlab Build Status
    :target: https://gitlab.tiker.net/inducer/pycuda/commits/main
.. image:: https://badge.fury.io/py/pycuda.png
    :target: https://pypi.org/project/pycuda
.. image:: https://zenodo.org/badge/1575319.svg
    :alt: Zenodo DOI for latest release
    :target: https://zenodo.org/badge/latestdoi/1575319

PyCUDA lets you access `Nvidia <https://nvidia.com>`_'s `CUDA
<https://nvidia.com/cuda/>`_ parallel computation API from Python.
Several wrappers of the CUDA API already exist-so what's so special
about PyCUDA?

* Object cleanup tied to lifetime of objects. This idiom, often
  called
  `RAII <https://en.wikipedia.org/wiki/Resource_Acquisition_Is_Initialization>`_
  in C++, makes it much easier to write correct, leak- and
  crash-free code. PyCUDA knows about dependencies, too, so (for
  example) it won't detach from a context before all memory
  allocated in it is also freed.

* Convenience. Abstractions like pycuda.driver.SourceModule and
  pycuda.gpuarray.GPUArray make CUDA programming even more
  convenient than with Nvidia's C-based runtime.

* Completeness. PyCUDA puts the full power of CUDA's driver API at
  your disposal, if you wish. It also includes code for
  interoperability with OpenGL.

* Automatic Error Checking. All CUDA errors are automatically
  translated into Python exceptions.

* Speed. PyCUDA's base layer is written in C++, so all the niceties
  above are virtually free.

* Helpful `Documentation <https://documen.tician.de/pycuda>`_.

Relatedly, like-minded computing goodness for `OpenCL <https://www.khronos.org/registry/OpenCL/>`_
is provided by PyCUDA's sister project `PyOpenCL <https://pypi.org/project/pyopencl>`_.
