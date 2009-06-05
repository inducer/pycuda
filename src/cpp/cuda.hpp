// A C++ wrapper for CUDA




#ifndef _AFJDFJSDFSD_PYCUDA_HEADER_SEEN_CUDA_HPP
#define _AFJDFJSDFSD_PYCUDA_HEADER_SEEN_CUDA_HPP




#include <cuda.h>
#include <stdexcept>
#include <boost/shared_ptr.hpp>
#include <boost/foreach.hpp>
#include <boost/weak_ptr.hpp>
#include <utility>
#include <stack>
#include <iostream>
#include <vector>
#include <boost/python.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/tss.hpp>
#include <boost/version.hpp>

#if (BOOST_VERSION/100) < 1035
#warning *****************************************************************
#warning **** Your version of Boost C++ is likely too old for PyCUDA. ****
#warning *****************************************************************
#endif




// #define CUDAPP_TRACE_CUDA




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
#define CUDAPP_CATCH_WARN_OOT_LEAK(TYPE) \
  catch (cuda::cannot_activate_out_of_thread_context) \
  { }
  // In all likelihood, this TYPE's managing thread has exited, and
  // therefore its context has already been deleted. No need to harp
  // on the fact that we still thought there was cleanup to do.

  // std::cerr << "PyCUDA WARNING: leaked out-of-thread " #TYPE " instance" << std::endl; */



namespace cuda
{
  namespace py = boost::python;




  class error : public std::runtime_error
  {
    private:
      const char *m_routine;
      CUresult m_code;

    private:
      static std::string make_message(const char *rout, CUresult c, const char *msg)
      {
        std::string result = rout;
        result += " failed: ";
        result += curesult_to_str(c);
        if (msg)
        {
          result += " - ";
          result += msg;
        }
        return result;
      }

    public:
      error(const char *rout, CUresult c, const char *msg=0)
        : std::runtime_error(make_message(rout, c, msg)),
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

  struct cannot_activate_out_of_thread_context : public std::logic_error
  { 
    cannot_activate_out_of_thread_context(std::string const &w)
      : std::logic_error(w)
    { }
  };




  // version query ------------------------------------------------------------
#if CUDA_VERSION >= 2020
  inline int get_driver_version()
  {
    int result;
    CUDAPP_CALL_GUARDED(cuDriverGetVersion, (&result));
    return result;
  }
#endif




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

      int get_attribute(CUdevice_attribute attr) const
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

      CUdevice handle() const
      { return m_device; }

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
  /* A word on context management: We don't let CUDA's context stack get more
   * than one deep. CUDA only supports pushing floating contexts. We may wish
   * to push contexts that are already active at a deeper stack level, so we
   * maintain all contexts floating other than the top one.
   */

  // for friend decl
  namespace gl { 
    boost::shared_ptr<context> 
        make_gl_context(device const &dev, unsigned int flags);
  }

  typedef std::stack<boost::weak_ptr<context>,
    std::vector<boost::weak_ptr<context> > > context_stack_t;
  extern boost::thread_specific_ptr<context_stack_t> context_stack_ptr;

  inline context_stack_t &get_context_stack()
  {
    if (context_stack_ptr.get() == 0)
      context_stack_ptr.reset(new context_stack_t);

    return *context_stack_ptr;
  }

  class context : boost::noncopyable
  {
    private:
      CUcontext m_context;
      bool m_valid;
      unsigned m_use_count;
      boost::thread::id m_thread;

    public:
      context(CUcontext ctx)
        : m_context(ctx), m_valid(true), m_use_count(1), 
        m_thread(boost::this_thread::get_id())
      { }

      ~context()
      { 
        if (m_valid)
        {
          if (m_use_count)
            std::cerr 
              << "-----------------------------------------------------------" << std::endl
              << "PyCUDA WARNING: I'm being asked to destroy a " << std::endl
              << "context that's part of the current context stack." << std::endl
              << "-----------------------------------------------------------" << std::endl
              << "I will pick the next lower active context from the" << std::endl
              << "context stack. Since this choice is happening" << std::endl
              << "at an unspecified point in time, your code" << std::endl
              << "may be making false assumptions about which" << std::endl
              << "context is active at what point." << std::endl
              << "Call Context.pop() to avoid this warning." << std::endl
              << "-----------------------------------------------------------" << std::endl
              << "If Python is terminating abnormally (eg. exiting upon an" << std::endl
              << "unhandled exception), you may ignore this." << std::endl
              << "-----------------------------------------------------------" << std::endl;
          detach();
        }
      }

      CUcontext handle() const
      { return m_context; }

      boost::thread::id thread_id() const
      { return m_thread; }

      void detach()
      {
        if (m_valid)
        {
          if (current_context().get() == this)
          {
            CUDAPP_CALL_GUARDED(cuCtxDetach, (m_context));
          }
          else
          {
            if (m_thread == boost::this_thread::get_id())
            {
              CUDAPP_CALL_GUARDED(cuCtxDestroy, (m_context));
            }
            else
            {
              // In all likelihood, this context's managing thread has exited, and 
              // therefore this context has already been deleted. No need to harp
              // on the fact that we still thought there was cleanup to do.

              // std::cerr << "PyCUDA WARNING: leaked out-of-thread context " << std::endl;
            }
          }

          m_valid = false;

          boost::shared_ptr<context> new_active = current_context(this);
          if (new_active.get())
            CUDAPP_CALL_GUARDED(cuCtxPushCurrent, (new_active->m_context)); 
        }
        else
          throw error("context::detach", CUDA_ERROR_INVALID_CONTEXT,
              "cannot detach from invalid context");
      }

      static device get_device()
      { 
        CUdevice dev;
        CUDAPP_CALL_GUARDED(cuCtxGetDevice, (&dev)); 
        return device(dev);
      }

#if CUDA_VERSION >= 2000

      static void prepare_context_switch()
      {
        if (get_context_stack().size())
        {
          CUcontext popped;
          CUDAPP_CALL_GUARDED(cuCtxPopCurrent, (&popped)); 
        }
      }

      void pop()
      { 
        prepare_context_switch();
        get_context_stack().pop();
        --m_use_count;

        boost::shared_ptr<context> current = current_context();
        if (current)
          CUDAPP_CALL_GUARDED(cuCtxPushCurrent, (current_context()->m_context)); 
      }
#else
      static void prepare_context_switch() { }
#endif

      static void synchronize()
      { CUDAPP_CALL_GUARDED_THREADED(cuCtxSynchronize, ()); }

      static boost::shared_ptr<context> current_context(context *except=0)
      {
        while (true)
        {
          if (get_context_stack().size() == 0)
            return boost::shared_ptr<context>();
          boost::weak_ptr<context> result(get_context_stack().top());
          if (!result.expired() && result.lock().get() != except)
          {
            // good, weak pointer didn't expire
            // (treating except as expired weak pointer)
            return result.lock();
          }
          else
          {
            // weak pointer invalidated, pop it and try again.
            get_context_stack().pop();
          }
        }
      }

      friend class device;
      friend void context_push(boost::shared_ptr<context> ctx);
      friend boost::shared_ptr<context> 
          gl::make_gl_context(device const &dev, unsigned int flags);
  };




  inline
  boost::shared_ptr<context> device::make_context(unsigned int flags)
  {
    context::prepare_context_switch();

    CUcontext ctx;
    CUDAPP_CALL_GUARDED(cuCtxCreate, (&ctx, flags, m_device));
    boost::shared_ptr<context> result(new context(ctx));
    get_context_stack().push(result);
    return result;
  }




#if CUDA_VERSION >= 2000
  inline
  void context_push(boost::shared_ptr<context> ctx)
  { 
    context::prepare_context_switch();

    CUDAPP_CALL_GUARDED(cuCtxPushCurrent, (ctx->m_context)); 
    get_context_stack().push(ctx);
    ++ctx->m_use_count;
  }
#endif




  class explicit_context_dependent
  {
    private:
      boost::shared_ptr<context> m_ward_context;

    public:
      void acquire_context()
      {
        m_ward_context = context::current_context();
        if (m_ward_context.get() == 0)
          throw error("explicit_context_dependent",
              CUDA_ERROR_INVALID_CONTEXT,
              "no currently active context?");
      }

      void release_context()
      {
        m_ward_context.reset();
      }

      boost::shared_ptr<context> get_context()
      {
        return m_ward_context;
      }
  };

  class context_dependent : public explicit_context_dependent
  {
    private:
      boost::shared_ptr<context> m_ward_context;

    public:
      context_dependent()
      { acquire_context(); }
  };


  class scoped_context_activation
  {
    private:
      boost::shared_ptr<context> m_context;
      bool m_did_switch;

    public:
      scoped_context_activation(boost::shared_ptr<context> ctx)
        : m_context(ctx)
      { 
        m_did_switch = context::current_context() != m_context;
        if (m_did_switch)
        {
          if (boost::this_thread::get_id() != m_context->thread_id())
            throw cuda::cannot_activate_out_of_thread_context(
                "cannot activate out-of-thread context");
#if CUDA_VERSION >= 2000
          context_push(m_context);
#else
          throw cuda::error("scoped_context_activation", CUDA_ERROR_INVALID_CONTEXT,
              "not available in CUDA < 2.0");
#endif
        }
      }

      ~scoped_context_activation()
      {
#if CUDA_VERSION >= 2000
        if (m_did_switch)
          m_context->pop();
#endif
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
      { 
        try
        {
          scoped_context_activation ca(get_context());
          CUDAPP_CALL_GUARDED(cuStreamDestroy, (m_stream)); 
        }
        CUDAPP_CATCH_WARN_OOT_LEAK(stream);
      }

      void synchronize()
      { CUDAPP_CALL_GUARDED_THREADED(cuStreamSynchronize, (m_stream)); }

      CUstream handle() const
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
      { free(); }

      void free()
      {
        if (m_managed)
        {
          try
          {
            scoped_context_activation ca(get_context());
            CUDAPP_CALL_GUARDED(cuArrayDestroy, (m_array)); 
          }
          CUDAPP_CATCH_WARN_OOT_LEAK(array);

          m_managed = false;
          release_context();
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

      CUarray handle() const
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

      CUtexref handle() const
      { return m_texref; }

      void set_array(boost::shared_ptr<array> ary)
      { 
        CUDAPP_CALL_GUARDED(cuTexRefSetArray, (m_texref, 
            ary->handle(), CU_TRSA_OVERRIDE_FORMAT)); 
        m_array = ary;
      }

      unsigned int set_address(CUdeviceptr dptr, unsigned int bytes, bool allow_offset=false)
      { 
        unsigned int byte_offset;
        CUDAPP_CALL_GUARDED(cuTexRefSetAddress, (&byte_offset,
              m_texref, dptr, bytes)); 

        if (!allow_offset && byte_offset != 0)
          throw cuda::error("texture_reference::set_address", CUDA_ERROR_INVALID_VALUE,
              "texture binding resulted in offset, but allow_offset was false");

        m_array.reset();
        return byte_offset;
      }

#if CUDA_VERSION >= 2020
      void set_address_2d(CUdeviceptr dptr, 
          const CUDA_ARRAY_DESCRIPTOR &descr, unsigned int pitch)
      { 
        CUDAPP_CALL_GUARDED(cuTexRefSetAddress2D, (m_texref, &descr, dptr, pitch)); 
      }
#endif

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
        try
        {
          scoped_context_activation ca(get_context());
          CUDAPP_CALL_GUARDED(cuModuleUnload, (m_module));
        }
        CUDAPP_CATCH_WARN_OOT_LEAK(module);
      }

      CUmodule handle() const
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
    CUDAPP_CALL_GUARDED(cuModuleGetTexRef, (&tr, mod->handle(), name));
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
            CU_PARAM_TR_DEFAULT, tr.handle())); 
      }

      void launch()
      { CUDAPP_CALL_GUARDED_THREADED(cuLaunch, (m_function)); }
      void launch_grid(int grid_width, int grid_height)
      { CUDAPP_CALL_GUARDED_THREADED(cuLaunchGrid, (m_function, grid_width, grid_height)); }
      void launch_grid_async(int grid_width, int grid_height, const stream &s)
      { CUDAPP_CALL_GUARDED_THREADED(cuLaunchGridAsync, (m_function, grid_width, grid_height, s.handle())); }

#if CUDA_VERSION >= 2020
      int get_attribute(CUfunction_attribute attr) const
      {
        int result;
        CUDAPP_CALL_GUARDED(cuFuncGetAttribute, (&result, attr, m_function));
        return result;
      }
#endif
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
          try
          {
            scoped_context_activation ca(get_context());
            mem_free(m_devptr);
          }
          CUDAPP_CATCH_WARN_OOT_LEAK(device_allocation);

          release_context();
          m_valid = false;
        }
        else
          throw cuda::error("device_allocation::free", CUDA_ERROR_INVALID_HANDLE);
      }

      ~device_allocation()
      {
        if (m_valid)
          free();
      }
      
      operator CUdeviceptr() const
      { return m_devptr; }
  };

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
  { CUDAPP_CALL_GUARDED_THREADED(cuMemcpyDtoA, (ary.handle(), index, src, len)); }

  inline
  void memcpy_atod(CUdeviceptr dst, array const &ary, unsigned int index, unsigned int len)
  { CUDAPP_CALL_GUARDED_THREADED(cuMemcpyAtoD, (dst, ary.handle(), index, len)); }

  inline
  void memcpy_atoa(
      array const &dst, unsigned int dst_index, 
      array const &src, unsigned int src_index, 
      unsigned int len)
  { CUDAPP_CALL_GUARDED_THREADED(cuMemcpyAtoA, (dst.handle(), dst_index, src.handle(), src_index, len)); }




  // structured memcpy --------------------------------------------------------
#if PY_VERSION_HEX >= 0x02050000
  typedef Py_ssize_t PYCUDA_BUFFER_SIZE_T;
#else
  typedef int PYCUDA_BUFFER_SIZE_T;
#endif

#define MEMCPY_SETTERS \
    void set_src_host(py::object buf_py) \
    { \
      srcMemoryType = CU_MEMORYTYPE_HOST; \
      PYCUDA_BUFFER_SIZE_T len; \
      if (PyObject_AsReadBuffer(buf_py.ptr(), &srcHost, &len)) \
        throw py::error_already_set(); \
    } \
    \
    void set_src_array(array const &ary)  \
    {  \
      srcMemoryType = CU_MEMORYTYPE_ARRAY; \
      srcArray = ary.handle();  \
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
      PYCUDA_BUFFER_SIZE_T len; \
      if (PyObject_AsWriteBuffer(buf_py.ptr(), &dstHost, &len)) \
        throw py::error_already_set(); \
    } \
    \
    void set_dst_array(array const &ary) \
    { \
      dstMemoryType = CU_MEMORYTYPE_ARRAY; \
      dstArray = ary.handle(); \
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
    { CUDAPP_CALL_GUARDED_THREADED(cuMemcpy2DAsync, (this, s.handle())); }
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
      srcLOD = 0;

      dstXInBytes = 0;
      dstY = 0;
      dstZ = 0;
      dstLOD = 0;
    }

    MEMCPY_SETTERS;

    void execute() const
    {
      CUDAPP_CALL_GUARDED_THREADED(cuMemcpy3D, (this));
    }

    void execute_async(const stream &s) const
    { CUDAPP_CALL_GUARDED_THREADED(cuMemcpy3DAsync, (this, s.handle())); }
  };
#endif


  // host memory --------------------------------------------------------------
  inline void *mem_alloc_host(unsigned int size, unsigned flags=0)
  {
    void *m_data;
#if CUDA_VERSION >= 2020
    CUDAPP_CALL_GUARDED(cuMemHostAlloc, (&m_data, size, flags));
#else
    if (flags != 0)
      throw cuda::error("mem_alloc_host", CUDA_ERROR_INVALID_VALUE,
          "nonzero flags in mem_alloc_host not allowed in CUDA 2.1 and older");
    CUDAPP_CALL_GUARDED(cuMemAllocHost, (&m_data, size));
#endif
    return m_data;
  }

  inline void mem_free_host(void *ptr)
  {
    CUDAPP_CALL_GUARDED(cuMemFreeHost, (ptr));
  }




  struct host_allocation : public boost::noncopyable, public context_dependent
  {
    private:
      bool m_valid;
      void *m_data;

    public:
      host_allocation(unsigned bytesize, unsigned flags=0)
        : m_valid(true), m_data(mem_alloc_host(bytesize, flags))
      { }

      ~host_allocation()
      { 
        if (m_valid)
          free(); 
      }

      void free()
      {
        if (m_valid)
        {
          try
          {
            scoped_context_activation ca(get_context());
            mem_free_host(m_data); 
          }
          CUDAPP_CATCH_WARN_OOT_LEAK(host_allocation);

          release_context();
          m_valid = false;
        }
        else
          throw cuda::error("host_allocation::free", CUDA_ERROR_INVALID_HANDLE);
      }
      
      void *data()
      { return m_data; }

#if CUDA_VERSION >= 2020
      CUdeviceptr get_device_pointer()
      {
        CUdeviceptr result;
        CUDAPP_CALL_GUARDED(cuMemHostGetDevicePointer, (&result, m_data, 0));
        return result;
      }
#endif

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
      { 
        try
        {
          scoped_context_activation ca(get_context());
          CUDAPP_CALL_GUARDED(cuEventDestroy, (m_event)); 
        }
        CUDAPP_CATCH_WARN_OOT_LEAK(event);
      }

      void record()
      { CUDAPP_CALL_GUARDED(cuEventRecord, (m_event, 0)); }

      void record_in_stream(stream const &str)
      { CUDAPP_CALL_GUARDED(cuEventRecord, (m_event, str.handle())); }

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
