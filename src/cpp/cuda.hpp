// A C++ wrapper for CUDA (not quite yet)




#ifndef _AFJDFJSDFSD_PYCUDA_HEADER_SEEN_CUDA_HPP
#define _AFJDFJSDFSD_PYCUDA_HEADER_SEEN_CUDA_HPP




#include <cuda.h>
#include <stdexcept>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <utility>
#include <stack>
#include <iostream>
#include <vector>
#include <boost/python.hpp>




//#define CUDAPP_TRACE_CUDA




#ifdef CUDAPP_TRACE_CUDA
  #define CUDAPP_PRINT_CALL_TRACE(NAME) std::cerr << NAME << std::endl;
#else
  #define CUDAPP_PRINT_CALL_TRACE(NAME) /*nothing*/
#endif

#define CUDAPP_CALL_GUARDED_THREADED(NAME, ARGLIST) \
  { \
    CUDAPP_PRINT_CALL_TRACE(#NAME); \
    CUresult cu_status_code; \
    Py_BEGIN_ALLOW_THREADS \
      cu_status_code = NAME ARGLIST; \
    Py_END_ALLOW_THREADS \
    if (cu_status_code != CUDA_SUCCESS) \
      throw cuda::error(#NAME, cu_status_code);\
  }

#define CUDAPP_CALL_GUARDED(NAME, ARGLIST) \
  { \
    CUDAPP_PRINT_CALL_TRACE(#NAME); \
    CUresult cu_status_code; \
    cu_status_code = NAME ARGLIST; \
    if (cu_status_code != CUDA_SUCCESS) \
      throw cuda::error(#NAME, cu_status_code);\
  }



namespace cuda
{
  namespace py = boost::python;




  class error : public std::runtime_error
  {
    private:
      const char *m_routine;
      CUresult m_code;

    public:
      error(const char *rout, CUresult c)
        : std::runtime_error(
            rout + std::string(" failed: ") + curesult_to_str(c)),
        m_routine(rout), m_code(c)
      { }

      const char *routine() const
      {
        return m_routine;
      }

      CUresult code() const
      {
        return m_code;
      }

      static const char *curesult_to_str(CUresult e)
      {
        switch (e)
        {
          case CUDA_SUCCESS: return "success";
          case CUDA_ERROR_INVALID_VALUE: return "invalid value";
          case CUDA_ERROR_OUT_OF_MEMORY: return "out of memory";
          case CUDA_ERROR_NOT_INITIALIZED: return "not initialized";
#if CUDA_VERSION >= 2000
          case CUDA_ERROR_DEINITIALIZED: return "deinitialized";
#endif

          case CUDA_ERROR_NO_DEVICE: return "no device";
          case CUDA_ERROR_INVALID_DEVICE: return "invalid device";

          case CUDA_ERROR_INVALID_IMAGE: return "invalid image";
          case CUDA_ERROR_INVALID_CONTEXT: return "invalid context";
          case CUDA_ERROR_CONTEXT_ALREADY_CURRENT: return "context already current";
          case CUDA_ERROR_MAP_FAILED: return "map failed";
          case CUDA_ERROR_UNMAP_FAILED: return "unmap failed";
          case CUDA_ERROR_ARRAY_IS_MAPPED: return "array is mapped";
          case CUDA_ERROR_ALREADY_MAPPED: return "already mapped";
          case CUDA_ERROR_NO_BINARY_FOR_GPU: return "no binary for gpu";
          case CUDA_ERROR_ALREADY_ACQUIRED: return "already acquired";
          case CUDA_ERROR_NOT_MAPPED: return "not mapped";

          case CUDA_ERROR_INVALID_SOURCE: return "invalid source";
          case CUDA_ERROR_FILE_NOT_FOUND: return "file not found";

          case CUDA_ERROR_INVALID_HANDLE: return "invalid handle";

          case CUDA_ERROR_NOT_FOUND: return "not found";

          case CUDA_ERROR_NOT_READY: return "not ready";

          case CUDA_ERROR_LAUNCH_FAILED: return "launch failed";
          case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: return "launch out of resources";
          case CUDA_ERROR_LAUNCH_TIMEOUT: return "launch timeout";
          case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: return "launch incompatible texturing";

          case CUDA_ERROR_UNKNOWN: return "unknown";

          default: return "invalid error code";
        }
      }
  };




  // device -------------------------------------------------------------------
  class context;

  class device
  {
    private:
      CUdevice m_device;

    public:
      device(CUdevice dev)
        : m_device(dev)
      { }

      static int count()
      {
        int result;
        CUDAPP_CALL_GUARDED(cuDeviceGetCount, (&result));
        return result;
      }

      std::string name()
      {
        char buffer[1024];
        CUDAPP_CALL_GUARDED(cuDeviceGetName, (buffer, sizeof(buffer), m_device));
        return buffer;
      }

      py::tuple compute_capability()
      {
        int major, minor;
        CUDAPP_CALL_GUARDED(cuDeviceComputeCapability, (&major, &minor, m_device));
        return py::make_tuple(major, minor);
      }

      unsigned int total_memory()
      {
        unsigned int bytes;
        CUDAPP_CALL_GUARDED(cuDeviceTotalMem, (&bytes, m_device));
        return bytes;
      }

      int get_attribute(CUdevice_attribute attr)
      {
        int result;
        CUDAPP_CALL_GUARDED(cuDeviceGetAttribute, (&result, attr, m_device));
        return result;
      }

      bool operator==(const device &other) const
      {
        return m_device == other.m_device;
      }

      bool operator!=(const device &other) const
      {
        return m_device != other.m_device;
      }

      long hash() const
      {
        return m_device;
      }

      boost::shared_ptr<context> make_context(unsigned int flags);
  };

  inline
  void init(unsigned int flags) 
  { 
    CUDAPP_CALL_GUARDED(cuInit, (flags)); 
  }

  inline
  device *make_device(int ordinal)
  { 
    CUdevice result;
    CUDAPP_CALL_GUARDED(cuDeviceGet, (&result, ordinal)); 
    return new device(result);
  }




  // context ------------------------------------------------------------------
  class context
  {
    private:
      CUcontext m_context;
      bool m_valid;
      typedef std::stack<boost::weak_ptr<context>,
        std::vector<boost::weak_ptr<context> > > context_stack_t;
      
      static context_stack_t m_context_stack;

    public:
      context(CUcontext ctx, bool borrowed)
        : m_context(ctx), m_valid(!borrowed)
      { 
        if (borrowed)
        {
          CUDAPP_CALL_GUARDED(cuCtxAttach, (&m_context, 0));
          m_valid = true;
        }
      }

      context(context const &src)
        : m_context(src.m_context), m_valid(false)
      { 
        CUDAPP_CALL_GUARDED(cuCtxAttach, (&m_context, 0));
        m_valid = true;
      }

      context &operator=(const context &src)
      {
        detach();
        m_context = src.m_context;
        CUDAPP_CALL_GUARDED(cuCtxAttach, (&m_context, 0));
        m_valid = true;
      }

      ~context()
      { detach(); }

      void detach()
      {
        if (m_valid)
        {
          CUDAPP_CALL_GUARDED(cuCtxDetach, (m_context));
          m_valid = false;
          m_context_stack.pop();
        }
      }

#if CUDA_VERSION >= 2000
      void pop()
      { 
        CUcontext popped;
        CUDAPP_CALL_GUARDED(cuCtxPopCurrent, (&popped)); 
        if (popped != m_context)
          throw std::runtime_error("popped the wrong context");
        m_context_stack.pop();
      }

      static device get_device()
      { 
        CUdevice dev;
        CUDAPP_CALL_GUARDED(cuCtxGetDevice, (&dev)); 
        return device(dev);
      }
#endif

      static void synchronize()
      { CUDAPP_CALL_GUARDED_THREADED(cuCtxSynchronize, ()); }

      static boost::shared_ptr<context> current_context()
      {
        return boost::shared_ptr<context>(m_context_stack.top());
      }

      friend class device;
      friend void context_push(boost::shared_ptr<context> ctx);
  };




  inline
  boost::shared_ptr<context> device::make_context(unsigned int flags)
  {
    CUcontext ctx;
    CUDAPP_CALL_GUARDED(cuCtxCreate, (&ctx, flags, m_device));
    boost::shared_ptr<context> result(new context(ctx, false));
    context::m_context_stack.push(result);
    return result;
  }

#if CUDA_VERSION >= 2000
  inline
  void context_push(boost::shared_ptr<context> ctx)
  { 
    CUDAPP_CALL_GUARDED(cuCtxPushCurrent, (ctx->m_context)); 
    context::m_context_stack.push(ctx);
  }
#endif




  class context_dependent
  {
    private:
      boost::shared_ptr<context> m_ward_context;

    public:
      context_dependent()
        : m_ward_context(context::current_context())
      { }
  };

  class explicit_context_dependent
  {
    private:
      boost::shared_ptr<context> m_ward_context;

    public:
      void acquire_context()
      {
        m_ward_context = context::current_context();
      }

      void release_context()
      {
        m_ward_context = boost::shared_ptr<context>();
      }

  };

  // streams ------------------------------------------------------------------
  class stream : public boost::noncopyable, public context_dependent
  {
    private:
      CUstream m_stream;

    public:
      stream(unsigned int flags=0)
      { CUDAPP_CALL_GUARDED(cuStreamCreate, (&m_stream, flags)); }

      ~stream()
      { CUDAPP_CALL_GUARDED(cuStreamDestroy, (m_stream)); }

      void synchronize()
      { CUDAPP_CALL_GUARDED_THREADED(cuStreamSynchronize, (m_stream)); }

      CUstream data() const
      { return m_stream; }

      bool is_done() const
      { 
#ifdef TRACE_CUDA
        std::cerr << "cuStreamQuery" << std::endl;
#endif
        CUresult result = cuStreamQuery(m_stream);
        switch (result)
        {
          case CUDA_SUCCESS: 
            return true;
          case CUDA_ERROR_NOT_READY: 
            return false;
          default:
            throw error("cuStreamQuery", result);
        }
      }
  };




  // arrays -------------------------------------------------------------------
  class array : public boost::noncopyable, public context_dependent
  {
    private:
      CUarray m_array;
      bool m_managed;

    public:
      array(const CUDA_ARRAY_DESCRIPTOR &descr)
        : m_managed(true)
      { CUDAPP_CALL_GUARDED(cuArrayCreate, (&m_array, &descr)); }

#if CUDA_VERSION >= 2000
      array(const CUDA_ARRAY3D_DESCRIPTOR &descr)
        : m_managed(true)
      { CUDAPP_CALL_GUARDED(cuArray3DCreate, (&m_array, &descr)); }
#endif

      array(CUarray ary, bool managed)
        : m_array(ary), m_managed(managed)
      { }

      ~array()
      { 
        if (m_managed)
        {
          CUDAPP_CALL_GUARDED(cuArrayDestroy, (m_array)); 
        }
      }

      CUDA_ARRAY_DESCRIPTOR get_descriptor()
      {
        CUDA_ARRAY_DESCRIPTOR result;
        CUDAPP_CALL_GUARDED(cuArrayGetDescriptor, (&result, m_array));
        return result;
      }

#if CUDA_VERSION >= 2000
      CUDA_ARRAY3D_DESCRIPTOR get_descriptor_3d()
      {
        CUDA_ARRAY3D_DESCRIPTOR result;
        CUDAPP_CALL_GUARDED(cuArray3DGetDescriptor, (&result, m_array));
        return result;
      }
#endif

      CUarray data() const
      { return m_array; }
  };




  // texture reference --------------------------------------------------------
  class module;

  class texture_reference : public  boost::noncopyable
  {
    private:
      CUtexref m_texref;
      bool m_managed;

      // life support for array and module
      boost::shared_ptr<array> m_array;
      boost::shared_ptr<module> m_module;

    public:
      texture_reference()
        : m_managed(true)
      { CUDAPP_CALL_GUARDED(cuTexRefCreate, (&m_texref)); }

      texture_reference(CUtexref tr, bool managed)
        : m_texref(tr), m_managed(managed)
      { }

      ~texture_reference()
      { 
        if (m_managed)
        {
          CUDAPP_CALL_GUARDED(cuTexRefDestroy, (m_texref)); 
        }
      }

      void set_module(boost::shared_ptr<module> mod)
      { m_module = mod; }

      CUtexref data() const
      { return m_texref; }

      void set_array(boost::shared_ptr<array> ary)
      { 
        CUDAPP_CALL_GUARDED(cuTexRefSetArray, (m_texref, 
            ary->data(), CU_TRSA_OVERRIDE_FORMAT)); 
        m_array = ary;
      }

      unsigned int set_address(CUdeviceptr dptr, unsigned int bytes)
      { 
        unsigned int byte_offset;
        CUDAPP_CALL_GUARDED(cuTexRefSetAddress, (&byte_offset,
              m_texref, dptr, bytes)); 
        m_array.reset();
        return byte_offset;
      }

      void set_format(CUarray_format fmt, int num_packed_components)
      { CUDAPP_CALL_GUARDED(cuTexRefSetFormat, (m_texref, fmt, num_packed_components)); }

      void set_address_mode(int dim, CUaddress_mode am)
      { CUDAPP_CALL_GUARDED(cuTexRefSetAddressMode, (m_texref, dim, am)); }
      void set_filter_mode(CUfilter_mode fm)
      { CUDAPP_CALL_GUARDED(cuTexRefSetFilterMode, (m_texref, fm)); }

      void set_flags(unsigned int flags)
      { CUDAPP_CALL_GUARDED(cuTexRefSetFlags, (m_texref, flags)); }

      CUdeviceptr get_address()
      {
        CUdeviceptr result;
        CUDAPP_CALL_GUARDED(cuTexRefGetAddress, (&result, m_texref));
        return result;
      }
      array *get_array()
      {
        CUarray result;
        CUDAPP_CALL_GUARDED(cuTexRefGetArray, (&result, m_texref));
        return new array(result, false);
      }
      CUaddress_mode get_address_mode(int dim)
      {
        CUaddress_mode result;
        CUDAPP_CALL_GUARDED(cuTexRefGetAddressMode, (&result, m_texref, dim));
        return result;
      }
      CUfilter_mode get_filter_mode()
      {
        CUfilter_mode result;
        CUDAPP_CALL_GUARDED(cuTexRefGetFilterMode, (&result, m_texref));
        return result;
      }

#if CUDA_VERSION >= 2000
      py::tuple get_format()
      {
        CUarray_format fmt;
        int num_channels;
        CUDAPP_CALL_GUARDED(cuTexRefGetFormat, (&fmt, &num_channels, m_texref));
        return py::make_tuple(fmt, num_channels);
      }
#endif

      unsigned int get_flags()
      {
        unsigned int result;
        CUDAPP_CALL_GUARDED(cuTexRefGetFlags, (&result, m_texref));
        return result;
      }
  };




  // module -------------------------------------------------------------------
  class function;

  class module : public boost::noncopyable, public context_dependent
  {
    private:
      CUmodule m_module;

    public:
      module(CUmodule mod)
        : m_module(mod)
      { }

      ~module()
      {
        CUDAPP_CALL_GUARDED(cuModuleUnload, (m_module));
      }

      CUmodule data() const
      { return m_module; }

      function get_function(const char *name);
      py::tuple get_global(const char *name)
      {
        CUdeviceptr devptr;
        unsigned int bytes;
        CUDAPP_CALL_GUARDED(cuModuleGetGlobal, (&devptr, &bytes, m_module, name));
        return py::make_tuple(devptr, bytes);
      }
  };

  inline
  module *module_from_file(const char *filename)
  {
    CUmodule mod;
    CUDAPP_CALL_GUARDED(cuModuleLoad, (&mod, filename));
    return new module(mod);
  }

  inline
  texture_reference *module_get_texref(boost::shared_ptr<module> mod, const char *name)
  {
    CUtexref tr;
    CUDAPP_CALL_GUARDED(cuModuleGetTexRef, (&tr, mod->data(), name));
    std::auto_ptr<texture_reference> result(
        new texture_reference(tr, false));
    result->set_module(mod);
    return result.release();
  }




  // function -----------------------------------------------------------------
  class function
  {
    private:
      CUfunction m_function;

    public:
      function(CUfunction func)
        : m_function(func)
      { }

      void set_block_shape(int x, int y, int z)
      { CUDAPP_CALL_GUARDED(cuFuncSetBlockShape, (m_function, x, y, z)); }
      void set_shared_size(unsigned int bytes)
      { CUDAPP_CALL_GUARDED(cuFuncSetSharedSize, (m_function, bytes)); }

      void param_set_size(unsigned int bytes)
      { CUDAPP_CALL_GUARDED(cuParamSetSize, (m_function, bytes)); }
      void param_set(int offset, unsigned int value)
      { CUDAPP_CALL_GUARDED(cuParamSeti, (m_function, offset, value)); }
      void param_set(int offset, float value)
      { CUDAPP_CALL_GUARDED(cuParamSetf, (m_function, offset, value)); }
      void param_setv(int offset, void *buf, unsigned long len)
      { 
        CUDAPP_CALL_GUARDED(cuParamSetv, (m_function, offset, buf, len)); 
      }
      void param_set_texref(const texture_reference &tr)
      { 
        CUDAPP_CALL_GUARDED(cuParamSetTexRef, (m_function, 
            CU_PARAM_TR_DEFAULT, tr.data())); 
      }

      void launch()
      { CUDAPP_CALL_GUARDED_THREADED(cuLaunch, (m_function)); }
      void launch_grid(int grid_width, int grid_height)
      { CUDAPP_CALL_GUARDED_THREADED(cuLaunchGrid, (m_function, grid_width, grid_height)); }
      void launch_grid_async(int grid_width, int grid_height, const stream &s)
      { CUDAPP_CALL_GUARDED_THREADED(cuLaunchGridAsync, (m_function, grid_width, grid_height, s.data())); }
  };

  inline
  function module::get_function(const char *name)
  {
    CUfunction func;
    CUDAPP_CALL_GUARDED(cuModuleGetFunction, (&func, m_module, name));
    return function(func);
  }




  // device memory ------------------------------------------------------------
  inline
  py::tuple mem_get_info()
  {
    unsigned int free, total;
    CUDAPP_CALL_GUARDED(cuMemGetInfo, (&free, &total));
    return py::make_tuple(free, total);
  }

  inline 
  CUdeviceptr mem_alloc(unsigned long bytes)
  {
    CUdeviceptr devptr;
    CUDAPP_CALL_GUARDED(cuMemAlloc, (&devptr, bytes));
    return devptr;
  }

  inline 
  void mem_free(CUdeviceptr devptr)
  {
    CUDAPP_CALL_GUARDED(cuMemFree, (devptr));
  }

  class device_allocation : public boost::noncopyable, public context_dependent
  {
    private:
      bool m_valid;

    protected:
      CUdeviceptr m_devptr;

    public:
      device_allocation(CUdeviceptr devptr)
        : m_valid(true), m_devptr(devptr)
      { }

      void free()
      {
        if (m_valid)
        {
          mem_free(m_devptr);
          m_valid = false;
        }
        else
          throw cuda::error("device_allocation::free", CUDA_ERROR_INVALID_HANDLE);
      }

      ~device_allocation()
      {
        if (m_valid)
          mem_free(m_devptr);
      }
      
      operator CUdeviceptr() const
      { return m_devptr; }
  };

  inline 
  device_allocation *make_device_allocation(unsigned long bytes)
  {
    return new device_allocation(mem_alloc(bytes));
  }

  inline unsigned int mem_alloc_pitch(
      std::auto_ptr<device_allocation> &da,
        unsigned int width, unsigned int height, unsigned int access_size)
  {
    CUdeviceptr devptr;
    unsigned int pitch;
    CUDAPP_CALL_GUARDED(cuMemAllocPitch, (&devptr, &pitch, width, height, access_size));
    da = std::auto_ptr<device_allocation>(new device_allocation(devptr));
    return pitch;
  }

  inline
  py::tuple mem_get_address_range(CUdeviceptr ptr)
  {
    CUdeviceptr base;
    unsigned int size;
    CUDAPP_CALL_GUARDED(cuMemGetAddressRange, (&base, &size, ptr));
    return py::make_tuple(base, size);
  }

  // missing: htoa, atoh, dtoh, htod

  inline
  void memcpy_dtoa(array const &ary, unsigned int index, CUdeviceptr src, unsigned int len)
  { CUDAPP_CALL_GUARDED_THREADED(cuMemcpyDtoA, (ary.data(), index, src, len)); }

  inline
  void memcpy_atod(CUdeviceptr dst, array const &ary, unsigned int index, unsigned int len)
  { CUDAPP_CALL_GUARDED_THREADED(cuMemcpyAtoD, (dst, ary.data(), index, len)); }

  inline
  void memcpy_atoa(
      array const &dst, unsigned int dst_index, 
      array const &src, unsigned int src_index, 
      unsigned int len)
  { CUDAPP_CALL_GUARDED_THREADED(cuMemcpyAtoA, (dst.data(), dst_index, src.data(), src_index, len)); }




  // structured memcpy --------------------------------------------------------
#define MEMCPY_SETTERS \
    void set_src_host(py::object buf_py) \
    { \
      srcMemoryType = CU_MEMORYTYPE_HOST; \
      Py_ssize_t len; \
      if (PyObject_AsReadBuffer(buf_py.ptr(), &srcHost, &len)) \
        throw py::error_already_set(); \
    } \
    \
    void set_src_array(array const &ary)  \
    {  \
      srcMemoryType = CU_MEMORYTYPE_ARRAY; \
      srcArray = ary.data();  \
    } \
    \
    void set_src_device(CUdeviceptr devptr)  \
    { \
      srcMemoryType = CU_MEMORYTYPE_DEVICE; \
      srcDevice = devptr; \
    } \
    \
    void set_dst_host(py::object buf_py) \
    { \
      dstMemoryType = CU_MEMORYTYPE_HOST; \
      Py_ssize_t len; \
      if (PyObject_AsWriteBuffer(buf_py.ptr(), &dstHost, &len)) \
        throw py::error_already_set(); \
    } \
    \
    void set_dst_array(array const &ary) \
    { \
      dstMemoryType = CU_MEMORYTYPE_ARRAY; \
      dstArray = ary.data(); \
    } \
    \
    void set_dst_device(CUdeviceptr devptr)  \
    { \
      dstMemoryType = CU_MEMORYTYPE_DEVICE; \
      dstDevice = devptr; \
    }





  struct memcpy_2d : public CUDA_MEMCPY2D
  {
    memcpy_2d()
    {
      srcXInBytes = 0;
      srcY = 0;

      dstXInBytes = 0;
      dstY = 0;
    }

    MEMCPY_SETTERS;

    void execute(bool aligned) const
    {
      if (aligned)
      { CUDAPP_CALL_GUARDED_THREADED(cuMemcpy2D, (this)); }
      else
      { CUDAPP_CALL_GUARDED_THREADED(cuMemcpy2DUnaligned, (this)); }
    }

    void execute_async(const stream &s) const
    { CUDAPP_CALL_GUARDED_THREADED(cuMemcpy2DAsync, (this, s.data())); }
  };

#if CUDA_VERSION >= 2000
  struct memcpy_3d : public CUDA_MEMCPY3D
  {
    memcpy_3d()
    {
      reserved0 = 0;
      reserved1 = 0;

      srcXInBytes = 0;
      srcY = 0;
      srcZ = 0;

      dstXInBytes = 0;
      dstY = 0;
      dstZ = 0;
    }

    MEMCPY_SETTERS;

    void execute() const
    {
      CUDAPP_CALL_GUARDED_THREADED(cuMemcpy3D, (this));
    }

    void execute_async(const stream &s) const
    { CUDAPP_CALL_GUARDED_THREADED(cuMemcpy3DAsync, (this, s.data())); }
  };
#endif


  // host memory --------------------------------------------------------------
  struct host_allocation : public boost::noncopyable
  {
    private:
      void *m_data;

    public:
      host_allocation(unsigned bytesize)
        : m_data(0)
      { CUDAPP_CALL_GUARDED(cuMemAllocHost, (&m_data, bytesize)); }

      ~host_allocation()
      { CUDAPP_CALL_GUARDED(cuMemFreeHost, (m_data)); }
      
      void *data()
      { return m_data; }
  };




  // events -------------------------------------------------------------------
  class event : public boost::noncopyable, public context_dependent
  {
    private:
      CUevent m_event;

    public:
      event(unsigned int flags=0)
      { CUDAPP_CALL_GUARDED(cuEventCreate, (&m_event, flags)); }

      ~event()
      { CUDAPP_CALL_GUARDED(cuEventDestroy, (m_event)); }

      void record()
      { CUDAPP_CALL_GUARDED(cuEventRecord, (m_event, 0)); }

      void record_in_stream(stream const &str)
      { CUDAPP_CALL_GUARDED(cuEventRecord, (m_event, str.data())); }

      void synchronize()
      { CUDAPP_CALL_GUARDED_THREADED(cuEventSynchronize, (m_event)); }

      bool query() const
      { 
#ifdef TRACE_CUDA
        std::cerr << "cuEventQuery" << std::endl;
#endif
        CUresult result = cuEventQuery(m_event);
        switch (result)
        {
          case CUDA_SUCCESS: 
            return true;
          case CUDA_ERROR_NOT_READY: 
            return false;
          default:
            throw error("cuEventQuery", result);
        }
      }

      float time_since(event const &start)
      {
        float result;
        CUDAPP_CALL_GUARDED(cuEventElapsedTime, (&result, start.m_event, m_event));
        return result;
      }

      float time_till(event const &end)
      {
        float result;
        CUDAPP_CALL_GUARDED(cuEventElapsedTime, (&result, m_event, end.m_event));
        return result;
      }
  };
}




#endif
