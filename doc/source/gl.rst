.. _gl-interop:

OpenGL
======

.. module:: pycuda.gl

.. function :: make_context(dev, flags=0)

    Create and return a :class:`pycuda.driver.Context` that has GL interoperability
    enabled.

    .. warning ::

        This will fail with a rather unhelpful error message if you don't already 
        have a GL context created and active.

.. class :: graphics_map_flags

    Usage of OpenGL object from CUDA.

    .. attribute :: NONE

        Read and write access to mapped OpenGL object from CUDA code.

    .. attribute :: READ_ONLY

        Read only access to mapped OpenGL object from CUDA code.

    .. attribute :: WRITE_DISCARD

        Write only access to mapped OpenGL object from CUDA code. Reading
        is prohibited.

.. class :: RegisteredBuffer(bufobj, flags = CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)

    Object managing mapping of OpenGL buffers to CUDA. Cannot be used to
    map images.

    .. method :: gl_handle()
    .. method :: unregister()
    .. method :: map(stream=None)

        Return a :class:`RegisteredMapping`.

.. class :: RegisteredImage(bufobj, target, flags = CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)

    Object managing mapping of OpenGL textures and render buffers to CUDA.

    *target* must be be one of:

     * `GL_TEXTURE_2D`
     * `GL_TEXTURE_RECTANGLE`
     * `GL_TEXTURE_CUBE_MAP`
     * `GL_TEXTURE_3D`
     * `GL_TEXTURE_2D_ARRAY`
     * `GL_RENDERBUFFER`

    (see PyOpenGL docs)

    .. method :: gl_handle()
    .. method :: unregister()
    .. method :: map(stream=None)

        Return a :class:`RegisteredMapping`.

.. class :: RegisteredMapping

    .. method :: unmap(stream=None)

        If no stream is specified, the unmap will use the same stream as the original
        mapping.

    .. method :: device_ptr_and_size()

        Return a tuple *(dev_pointer, size)*.

        .. versionadded: 2011.1

    .. method :: array(index, level)

        Return an array for mapped image object for given array index and MIP level.

Automatic Initialization
------------------------

.. module:: pycuda.gl.autoinit

.. warning ::

    Importing :mod:`pycuda.gl.autoinit` will fail with a rather unhelpful error 
    message if you don't already have a GL context created and active.

.. data:: device
.. data:: context

Old-style (pre-CUDA 3.0) API
----------------------------

.. function :: init()

    Enable GL interoperability for the already-created (so far non-GL)
    and currently active :class:`pycuda.driver.Context`.

    According to the forum post referenced in the note below, this will succeed 
    on Windows XP and Linux, but it will not work on Windows Vista. There you 
    *have* to create the GL-enabled context using :func:`make_context`.

    .. warning ::

        This function is deprecated since CUDA 3.0 and PyCUDA 2011.1.

    .. warning ::

        This will fail with a rather unhelpful error message if you don't already 
        have a GL context created and active.

.. note ::

    See this `post <http://forums.nvidia.com/index.php?showtopic=88152>`_ on the
    Nvidia forums for a discussion of problems and solutions with the GL interop
    interface.

.. class :: BufferObject(bufobj)

    .. warning ::

        This class is deprecated since CUDA 3.0 and PyCUDA 2011.1.

    .. method :: unregister()
    .. attribute :: handle()
    .. method :: map()

.. class :: BufferObjectMapping

    .. warning ::

        This class is deprecated since CUDA 3.0 and PyCUDA 2011.1.
        It will be removed in PyCUDA 0.96.

    .. method :: unmap()
    .. method :: device_ptr()
    .. method :: size()

