.. highlight:: sh

Installation
============

This tutorial will walk you through the process of building PyCUDA. To follow,
you really only need four basic things:

* A UNIX-like machine with web access.
* `Nvidia <http://nvidia.com/>`_'s `CUDA <http://nvidia.com/cuda/>`_ toolkit.
  PyCUDA was developed against version 2.0 beta. It may work with other versions,
  too.
* A C++ compiler, preferably a Version 4.x gcc.
* A working `Python <http://www.python.org>`_ installation, Version 2.4 or newer.

Step 1: Install Boost
---------------------

You may already have a working copy of the `Boost C++ libraries
<http://www.boost.org>`_. If so, make sure that it's version 1.35.0 or newer.
If not, no problem, please follow this link to the simple `build and install instructions
<http://mathema.tician.de/software/install-boost>`_ that I wrote for Boost. 
Continue here when you're done.

Step 2: Download and unpack PyCUDA
-----------------------------------

`Download PyCUDA <http://pypi.python.org/pypi/pycuda>`_ and unpack it::

    $ tar xfz pycuda-VERSION.tar.gz

Step 3: Install Numpy
---------------------

PyCUDA is designed to work in conjunction with `numpy <http://numpy.org>`_,
Python's array package. 

Here's an easy way to install it, if you do not have it already::

    $ cd pycuda-VERSION
    $ su -c "python ez_setup.py" # this will install setuptools
    $ su -c "easy_install numpy" # this will install numpy using setuptools

(If you're not sure, repeating these commands will not hurt.)

Step 4: Build PyCUDA
--------------------

Next, just type::

    $ cd pycuda-VERSION # if you're not there already
    $ python configure.py \
      --boost-inc-dir=$HOME/pool/include/boost-1_35 \
      --boost-lib-dir=$HOME/pool/lib \
      --boost-python-libname=boost_python-gcc42-mt \
      --boost-thread-libname=boost_thread-gcc42-mt (0.93 and above only) \
      --cuda-root=/where/ever/you/installed/cuda
    $ su -c "make install"

Note that ``gcc42`` is a compiler tag that depends on the compiler
with which you built boost. Check the contents of your boost 
library directory to find out what the correct tag is. Also note that
you will (probably) have to change the value of :option:`--cuda-root`.

Once that works, congratulations! You've successfully built PyCUDA.

Step 5: Test PyCUDA
--------------------

If you'd like to be extra-careful, you can run PyCUDA's unit tests::

    $ cd pycuda-VERSION/test
    $ python test_driver.py

If it says "OK" at the end, you're golden.

Installing on Windows
---------------------

First, try running :command:`configure.py` as above.
If that fails, create a file called :file:`siteconf.py` containing the following, adapted
to match your system::

    BOOST_INC_DIR = [r'C:\Program Files\boost\boost_1_36_0']
    BOOST_LIB_DIR = [r'C:\Program Files\boost\boost_1_36_0\stage\lib']
    BOOST_PYTHON_LIBNAME = ['boost_python-mgw34']
    CUDA_ROOT = r'C:\CUDA'
    CUDADRV_LIB_DIR = [r'C:\CUDAlib']
    CUDADRV_LIBNAME = ['cuda']
    CXXFLAGS = []
    LDFLAGS = []

Subsequently, you may build and install PyCUDA by typing::

    $ python setup.py install
