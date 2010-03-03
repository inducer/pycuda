#ifndef _AFJDFJSDFSD_PYCUDA_HEADER_SEEN_CUDA_GL_HPP
#define _AFJDFJSDFSD_PYCUDA_HEADER_SEEN_CUDA_GL_HPP



#include <cuda.hpp>
#if defined(__APPLE__) || defined(MACOSX)
#include <OpenGL/gl.h>
#else  /* __APPLE__ */
#include <GL/gl.h>
#endif 

#include <cudaGL.h>


namespace cuda { namespace gl {
  inline
  void gl_init()
  {
    CUDAPP_CALL_GUARDED(cuGLInit, ());
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
      { CUDAPP_CALL_GUARDED(cuGLRegisterBufferObject, (handle)); }

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
          throw cuda::error("buffer_object::unregister", CUDA_ERROR_INVALID_HANDLE);
      }
  };



  class buffer_object_mapping : public context_dependent
  {
    private:
      boost::shared_ptr<buffer_object> m_buffer_object;
      CUdeviceptr m_devptr;
      unsigned int m_size;
      bool m_valid;

    public:
      buffer_object_mapping(
          boost::shared_ptr<buffer_object> bobj,
          CUdeviceptr devptr,
          unsigned int size)
        : m_buffer_object(bobj), m_devptr(devptr), m_size(size), m_valid(true)
      { }

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
          throw cuda::error("buffer_object_mapping::unmap", CUDA_ERROR_INVALID_HANDLE);
      }

      CUdeviceptr device_ptr() const
      { return m_devptr; }

      unsigned int size() const
      { return m_size; }
  };




  inline buffer_object_mapping *map_buffer_object(
      boost::shared_ptr<buffer_object> bobj)
  {
    CUdeviceptr devptr;
    unsigned int size;
    CUDAPP_CALL_GUARDED(cuGLMapBufferObject, (&devptr, &size, bobj->handle()));

    return new buffer_object_mapping(bobj, devptr, size);
  }
} }




#endif
