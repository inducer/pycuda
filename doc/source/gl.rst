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

        This function is deprecated since CUDA 3.0 and PyCUDA 0.95.

        This will fail with a rather unhelpful error message if you don't already 
        have a GL context created and active.

.. function :: make_context(dev, flags=0)

    Create and return a :class:`pycuda.driver.Context` that has GL interoperability
    enabled. Note that this is an *alternative* to calling :func:`init` on an 
    already-active context.

    .. warning ::

        This will fail with a rather unhelpful error message if you don't already 
        have a GL context created and active.

.. class :: map_flags

    Usage of OpenGL object from CUDA.

    .. attribute :: CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE

        Read and write access to mapped OpenGL object from CUDA code.

    .. attribute :: CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY

        Read only access to mapped OpenGL object from CUDA code.

    .. attribute :: CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD

        Write only access to mapped OpenGL object from CUDA code. Reading
        is prohibited.

.. class :: map_targets

    Type of OpenGL Image object that is mapped to CUDA.

    .. attribute :: GL_TEXTURE_2D
    .. attribute :: GL_TEXTURE_RECTANGLE
    .. attribute :: GL_TEXTURE_CUBE_MAP
    .. attribute :: GL_TEXTURE_3D
    .. attribute :: GL_TEXTURE_2D_ARRAY
    .. attribute :: GL_RENDERBUFFER

.. class :: BufferObject(bufobj)

    .. method :: unregister()
    .. method :: handle()
    .. method :: map()

    .. warning ::

        This class is deprecated since CUDA 3.0 and PyCUDA 0.95.
    
.. class :: BufferObjectMapping

    .. method :: unmap()
    .. method :: device_ptr()
    .. method :: size()

    .. warning ::

        This class is deprecated since CUDA 3.0 and PyCUDA 0.95.

.. class :: RegisteredBuffer(bufobj, flags = CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)

  Object managing mapping of OpenGL buffers to CUDA. Cannot be used to
  map images.

    .. method :: unregister()
    .. method :: handle()
    .. method :: map()
    
.. class :: RegisteredImage(bufobj, target, flags = CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)

  Object managing mapping of OpenGL textures and render buffers to CUDA.

    .. method :: unregister()
    .. method :: handle()
    .. method :: map()
    
.. class :: RegisteredMapping

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
