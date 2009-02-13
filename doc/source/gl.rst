GL Interoperability
===================

.. note::

    This functionality is scheduled for release 0.93 and only available in
    source-built versions of PyCuda's git tree.

.. module:: pycuda.gl

.. function :: init()
    
    Enable GL interoperability for the already-created (so far non-GL)
    and currently active :class:`pycuda.driver.Context`.

    According to the forum post referenced in the note below, this will succeed 
    on Windows XP and Linux, but it will not work on Windows Vista. There you 
    *have* to create the GL-enabled context using :func:`make_context`.

    .. warning ::

        This will fail with a rather unhelpful error message if you don't already 
        have a GL context created and active.

.. function :: make_context(dev, flags=0)

    Create and return a :class:`pycuda.driver.Context` that has GL interoperability
    enabled. Note that this is an *alternative* to calling :func:`init` on an 
    already-active context.

    .. warning ::

        This will fail with a rather unhelpful error message if you don't already 
        have a GL context created and active.

.. class :: BufferObject(bufobj)

    .. method :: unregister()
    .. method :: handle()
    .. method :: map()
    
.. class :: BufferObjectMapping

    .. method :: unmap()
    .. method :: device_ptr()
    .. method :: size()

.. note ::

    See this `post <http://forums.nvidia.com/index.php?showtopic=88152>`_ on the
    Nvidia forums for a discussion of problems and solutions with the GL interop
    interface.


Automatic Initialization
------------------------

.. module:: pycuda.gl.autoinit

.. warning ::

    Importing :mod:`pycuda.gl.autoinit` will fail with a rather unhelpful error 
    message if you don't already have a GL context created and active.

.. data:: device
.. data:: context
