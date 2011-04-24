#ifndef _AFJDFJSDFSD_PYCUDA_HEADER_SEEN_CUDA_GL_HPP
#define _AFJDFJSDFSD_PYCUDA_HEADER_SEEN_CUDA_GL_HPP



#include <cuda.hpp>
#if defined(__APPLE__) || defined(MACOSX)
#include <OpenGL/gl.h>
#else  /* __APPLE__ */
#include <GL/gl.h>
#endif 

#include <cudaGL.h>


namespace pycuda { namespace gl {

  // {{{ pre-3.0-style API

  inline
  void gl_init()
  {
    CUDAPP_CALL_GUARDED(cuGLInit, ());
    PyErr_Warn(
        PyExc_DeprecationWarning,
        "gl_init() has been deprecated since CUDA 3.0 "
        "and PyCUDA 2011.1.");
  }




  inline
  boost::shared_ptr<context> make_gl_context(device const &dev, unsigned int flags)
  {
    CUcontext ctx;
    CUDAPP_CALL_GUARDED(cuGLCtxCreate, (&ctx, flags, dev.handle()));
    boost::shared_ptr<context> result(new context(ctx));
    context_stack::get().push(result);
    return result;
  }




  class buffer_object : public context_dependent
  {
    private:
      GLuint m_handle;
      bool m_valid;

    public:
      buffer_object(GLuint handle)
        : m_handle(handle), m_valid(true)
      {
        CUDAPP_CALL_GUARDED(cuGLRegisterBufferObject, (handle));
        PyErr_Warn(
            PyExc_DeprecationWarning,
            "buffer_object has been deprecated since CUDA 3.0 "
            "and PyCUDA 2011.1.");
      }

      ~buffer_object()
      {
        if (m_valid)
          unregister();
      }

      GLuint handle()
      { return m_handle; }

      void unregister()
      {
        if (m_valid)
        {
          try
          {
            scoped_context_activation ca(get_context());
            CUDAPP_CALL_GUARDED_CLEANUP(cuGLUnregisterBufferObject, (m_handle));
            m_valid = false;
          }
          CUDAPP_CATCH_CLEANUP_ON_DEAD_CONTEXT(buffer_object);
        }
        else
          throw pycuda::error("buffer_object::unregister", CUDA_ERROR_INVALID_HANDLE);
      }
  };



  class buffer_object_mapping : public context_dependent
  {
    private:
      boost::shared_ptr<buffer_object> m_buffer_object;
      CUdeviceptr m_devptr;
      size_t m_size;
      bool m_valid;

    public:
      buffer_object_mapping(
          boost::shared_ptr<buffer_object> bobj,
          CUdeviceptr devptr,
          size_t size)
        : m_buffer_object(bobj), m_devptr(devptr), m_size(size), m_valid(true)
      { 
        PyErr_Warn(
            PyExc_DeprecationWarning,
            "buffer_object_mapping has been deprecated since CUDA 3.0 "
            "and PyCUDA 2011.1.");
      }

      ~buffer_object_mapping()
      {
        if (m_valid)
          unmap();
      }

      void unmap()
      {
        if (m_valid)
        {
          try
          {
            scoped_context_activation ca(get_context());
            CUDAPP_CALL_GUARDED_CLEANUP(cuGLUnmapBufferObject, (m_buffer_object->handle()));
            m_valid = false;
          }
          CUDAPP_CATCH_CLEANUP_ON_DEAD_CONTEXT(buffer_object_mapping)
        }
        else
          throw pycuda::error("buffer_object_mapping::unmap", CUDA_ERROR_INVALID_HANDLE);
      }

      CUdeviceptr device_ptr() const
      { return m_devptr; }

      size_t size() const
      { return m_size; }
  };




  inline buffer_object_mapping *map_buffer_object(
      boost::shared_ptr<buffer_object> bobj)
  {
    CUdeviceptr devptr;
    pycuda_size_t size;
    CUDAPP_CALL_GUARDED(cuGLMapBufferObject, (&devptr, &size, bobj->handle()));
    PyErr_Warn(
        PyExc_DeprecationWarning,
        "map_buffer_object has been deprecated since CUDA 3.0 "
        "and PyCUDA 2011.1.");

    return new buffer_object_mapping(bobj, devptr, size);
  }

  // }}}

  // {{{ new-style (3.0+) API

#if CUDAPP_CUDA_VERSION >= 3000
  class registered_object : public context_dependent
  {
    protected:
      GLuint m_gl_handle;
      bool m_valid;
      CUgraphicsResource m_resource;

    public:
      registered_object(GLuint gl_handle)
        : m_gl_handle(gl_handle), m_valid(true)
      {
      }

      ~registered_object()
      {
        if (m_valid)
          unregister();
      }

      GLuint gl_handle()
      { return m_gl_handle; }

      CUgraphicsResource resource()
      { return m_resource; }

      void unregister()
      {
        if (m_valid)
        {
          try
          {
            scoped_context_activation ca(get_context());
            CUDAPP_CALL_GUARDED_CLEANUP(
                cuGraphicsUnregisterResource, (m_resource));
            m_valid = false;
          }
          CUDAPP_CATCH_CLEANUP_ON_DEAD_CONTEXT(registered_object);
        }
        else
          throw pycuda::error("registered_object::unregister", 
              CUDA_ERROR_INVALID_HANDLE);
      }
  };

  class registered_buffer : public registered_object
  {
    public:
      registered_buffer(GLuint gl_handle, 
          CUgraphicsMapResourceFlags flags=CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)
        : registered_object(gl_handle)
      {
        CUDAPP_CALL_GUARDED(cuGraphicsGLRegisterBuffer, 
            (&m_resource, gl_handle, flags));
      }
  };

  class registered_image : public registered_object
  {
    public:
      registered_image(GLuint gl_handle, GLenum target, 
          CUgraphicsMapResourceFlags flags=CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)
        : registered_object(gl_handle)
      {
        CUDAPP_CALL_GUARDED(cuGraphicsGLRegisterImage, 
            (&m_resource, gl_handle, target, flags));
      }
  };



  class registered_mapping : public context_dependent
  {
    private:
      boost::shared_ptr<registered_object> m_object;
      boost::shared_ptr<stream> m_stream;
      bool m_valid;

    public:
      registered_mapping(
          boost::shared_ptr<registered_object> robj,
          boost::shared_ptr<stream> strm)
        : m_object(robj), m_stream(strm), m_valid(true)
      { }

      ~registered_mapping()
      {
        if (m_valid)
          unmap_no_strm();
      }

      void unmap_no_strm()
      {
        unmap(m_stream);
      }

      void unmap(boost::shared_ptr<stream> const &strm)
      {
        CUstream s_handle;
        if (!strm.get())
          s_handle = 0;
        else
          s_handle = strm->handle();

        if (m_valid)
        {
          try
          {
            scoped_context_activation ca(get_context());
            CUgraphicsResource res = m_object->resource();
            CUDAPP_CALL_GUARDED_CLEANUP(cuGraphicsUnmapResources,
                (1, &res, s_handle));
            m_valid = false;
          }
          CUDAPP_CATCH_CLEANUP_ON_DEAD_CONTEXT(registered_mapping)
        }
        else
          throw pycuda::error("registered_mapping::unmap", CUDA_ERROR_INVALID_HANDLE);
      }

      py::tuple device_ptr_and_size() const
      {
        CUdeviceptr devptr;
        pycuda_size_t size;
        CUDAPP_CALL_GUARDED(cuGraphicsResourceGetMappedPointer, 
            (&devptr, &size, m_object->resource()));
        return py::make_tuple(devptr, size);
      }

      inline
      pycuda::array *array(unsigned int index, unsigned int level) const
      {
        CUarray devptr;
        CUDAPP_CALL_GUARDED(cuGraphicsSubResourceGetMappedArray, 
            (&devptr, m_object->resource(), index, level));
        std::auto_ptr<pycuda::array> result(
            new pycuda::array(devptr, false));
        return result.release();
      }
  };




  inline registered_mapping *map_registered_object(
      boost::shared_ptr<registered_object> const &robj,
      py::object strm_py)
  {
    CUstream s_handle;
    boost::shared_ptr<stream> strm_sptr;

    if (strm_py.ptr() == Py_None)
    {
      s_handle = 0;
    }
    else
    {
      strm_sptr = py::extract<boost::shared_ptr<stream> >(strm_py);
      s_handle = strm_sptr->handle();
    }

    CUgraphicsResource res = robj->resource();
    CUDAPP_CALL_GUARDED(cuGraphicsMapResources,
        (1, &res, s_handle));

    return new registered_mapping(robj, strm_sptr);
  }
#endif

  // }}}

} }




#endif

// vim: foldmethod=marker
