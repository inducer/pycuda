// A C++ wrapper for CUDA




#ifndef _AFJDFJSDFSD_PYCUDA_HEADER_SEEN_CUDA_HPP
#define _AFJDFJSDFSD_PYCUDA_HEADER_SEEN_CUDA_HPP




// {{{ includes, configuration

#include <cuda.h>

#ifdef CUDAPP_PRETEND_CUDA_VERSION
#define CUDAPP_CUDA_VERSION CUDAPP_PRETEND_CUDA_VERSION
#else
#define CUDAPP_CUDA_VERSION CUDA_VERSION
#endif

#if CUDAPP_CUDA_VERSION >= 4000
#include <cudaProfiler.h>
#endif

#ifndef _MSC_VER
#include <stdint.h>
#endif
#include <stdexcept>
#include <boost/shared_ptr.hpp>
#include <boost/foreach.hpp>
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

// MAYBE? cuMemcpy, cuPointerGetAttribute
// TODO: cuCtxSetCurrent, cuCtxGetCurrent
// (use once the old, deprecated functions have been removed from CUDA)




// #define CUDAPP_TRACE_CUDA
#define CUDAPP_POST_30_BETA

#ifdef CUDAPP_PRETEND_CUDA_VERSION
#define CUDAPP_CUDA_VERSION CUDAPP_PRETEND_CUDA_VERSION
#else
#define CUDAPP_CUDA_VERSION CUDA_VERSION
#endif




#if (PY_VERSION_HEX < 0x02060000)
  #error PyCUDA does not support Python 2 versions earlier than 2.6.
#endif
#if (PY_VERSION_HEX >= 0x03000000) && (PY_VERSION_HEX < 0x03030000)
  #error PyCUDA does not support Python 3 versions earlier than 3.3.
#endif

typedef Py_ssize_t PYCUDA_BUFFER_SIZE_T;

// }}}


#define PYCUDA_PARSE_STREAM_PY \
    CUstream s_handle; \
    if (stream_py.ptr() != Py_None) \
    { \
      const stream &s = py::extract<const stream &>(stream_py); \
      s_handle = s.handle(); \
    } \
    else \
      s_handle = 0;



// {{{ tracing and error guards

#ifdef CUDAPP_TRACE_CUDA
  #define CUDAPP_PRINT_CALL_TRACE(NAME) \
    std::cerr << NAME << std::endl;
  #define CUDAPP_PRINT_CALL_TRACE_INFO(NAME, EXTRA_INFO) \
    std::cerr << NAME << " (" << EXTRA_INFO << ')' << std::endl;
  #define CUDAPP_PRINT_ERROR_TRACE(NAME, CODE) \
    if (CODE != CUDA_SUCCESS) \
      std::cerr << NAME << " failed with code " << CODE << std::endl;
#else
  #define CUDAPP_PRINT_CALL_TRACE(NAME) /*nothing*/
  #define CUDAPP_PRINT_CALL_TRACE_INFO(NAME, EXTRA_INFO) /*nothing*/
  #define CUDAPP_PRINT_ERROR_TRACE(NAME, CODE) /*nothing*/
#endif

#define CUDAPP_CALL_GUARDED_THREADED_WITH_TRACE_INFO(NAME, ARGLIST, TRACE_INFO) \
  { \
    CUDAPP_PRINT_CALL_TRACE_INFO(#NAME, TRACE_INFO); \
    CUresult cu_status_code; \
    Py_BEGIN_ALLOW_THREADS \
      cu_status_code = NAME ARGLIST; \
    Py_END_ALLOW_THREADS \
    if (cu_status_code != CUDA_SUCCESS) \
      throw pycuda::error(#NAME, cu_status_code);\
  }

#define CUDAPP_CALL_GUARDED_WITH_TRACE_INFO(NAME, ARGLIST, TRACE_INFO) \
  { \
    CUDAPP_PRINT_CALL_TRACE_INFO(#NAME, TRACE_INFO); \
    CUresult cu_status_code; \
    cu_status_code = NAME ARGLIST; \
    CUDAPP_PRINT_ERROR_TRACE(#NAME, cu_status_code); \
    if (cu_status_code != CUDA_SUCCESS) \
      throw pycuda::error(#NAME, cu_status_code);\
  }

#define CUDAPP_CALL_GUARDED_THREADED(NAME, ARGLIST) \
  { \
    CUDAPP_PRINT_CALL_TRACE(#NAME); \
    CUresult cu_status_code; \
    Py_BEGIN_ALLOW_THREADS \
      cu_status_code = NAME ARGLIST; \
    Py_END_ALLOW_THREADS \
    CUDAPP_PRINT_ERROR_TRACE(#NAME, cu_status_code); \
    if (cu_status_code != CUDA_SUCCESS) \
      throw pycuda::error(#NAME, cu_status_code);\
  }

#define CUDAPP_CALL_GUARDED(NAME, ARGLIST) \
  { \
    CUDAPP_PRINT_CALL_TRACE(#NAME); \
    CUresult cu_status_code; \
    cu_status_code = NAME ARGLIST; \
    CUDAPP_PRINT_ERROR_TRACE(#NAME, cu_status_code); \
    if (cu_status_code != CUDA_SUCCESS) \
      throw pycuda::error(#NAME, cu_status_code);\
  }
#define CUDAPP_CALL_GUARDED_CLEANUP(NAME, ARGLIST) \
  { \
    CUDAPP_PRINT_CALL_TRACE(#NAME); \
    CUresult cu_status_code; \
    cu_status_code = NAME ARGLIST; \
    CUDAPP_PRINT_ERROR_TRACE(#NAME, cu_status_code); \
    if (cu_status_code != CUDA_SUCCESS) \
      std::cerr \
        << "PyCUDA WARNING: a clean-up operation failed (dead context maybe?)" \
        << std::endl \
        << pycuda::error::make_message(#NAME, cu_status_code) \
        << std::endl; \
  }
#define CUDAPP_CATCH_CLEANUP_ON_DEAD_CONTEXT(TYPE) \
  catch (pycuda::cannot_activate_out_of_thread_context) \
  { } \
  catch (pycuda::cannot_activate_dead_context) \
  { \
    /* PyErr_Warn( \
        PyExc_UserWarning, #TYPE " in dead context was implicitly cleaned up");*/ \
  }
  // In all likelihood, this TYPE's managing thread has exited, and
  // therefore its context has already been deleted. No need to harp
  // on the fact that we still thought there was cleanup to do.

// }}}




namespace pycuda
{
  namespace py = boost::python;

  typedef
#if CUDAPP_CUDA_VERSION >= 3020
        size_t
#else
        unsigned int
#endif
        pycuda_size_t;

  typedef
#if defined(_WIN32) && defined(_WIN64)
    long long
#else
    long
#endif
    hash_type;



  // {{{ error reporting
  class error : public std::runtime_error
  {
    private:
      const char *m_routine;
      CUresult m_code;

    public:
      static std::string make_message(const char *rout, CUresult c, const char *msg=0)
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

      bool is_out_of_memory() const
      {
        return code() == CUDA_ERROR_OUT_OF_MEMORY;
      }

      static const char *curesult_to_str(CUresult e)
      {
#if CUDAPP_CUDA_VERSION >= 6000
        const char* errstr;
        cuGetErrorString(e, &errstr);
        return errstr;
#else
        switch (e)
        {
          case CUDA_SUCCESS: return "success";
          case CUDA_ERROR_INVALID_VALUE: return "invalid value";
          case CUDA_ERROR_OUT_OF_MEMORY: return "out of memory";
          case CUDA_ERROR_NOT_INITIALIZED: return "not initialized";
#if CUDAPP_CUDA_VERSION >= 2000
          case CUDA_ERROR_DEINITIALIZED: return "deinitialized";
#endif
#if CUDAPP_CUDA_VERSION >= 4000
          case CUDA_ERROR_PROFILER_DISABLED: return "profiler disabled";
          case CUDA_ERROR_PROFILER_NOT_INITIALIZED: return "profiler not initialized";
          case CUDA_ERROR_PROFILER_ALREADY_STARTED: return "profiler already started";
          case CUDA_ERROR_PROFILER_ALREADY_STOPPED: return "profiler already stopped";
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
#if CUDAPP_CUDA_VERSION >= 3000
          case CUDA_ERROR_NOT_MAPPED_AS_ARRAY: return "not mapped as array";
          case CUDA_ERROR_NOT_MAPPED_AS_POINTER: return "not mapped as pointer";
#ifdef CUDAPP_POST_30_BETA
          case CUDA_ERROR_ECC_UNCORRECTABLE: return "ECC uncorrectable";
#endif
#endif
#if CUDAPP_CUDA_VERSION >= 3010
          case CUDA_ERROR_UNSUPPORTED_LIMIT: return "unsupported limit";
#endif
#if CUDAPP_CUDA_VERSION >= 4000
          case CUDA_ERROR_CONTEXT_ALREADY_IN_USE: return "context already in use";
#endif

          case CUDA_ERROR_INVALID_SOURCE: return "invalid source";
          case CUDA_ERROR_FILE_NOT_FOUND: return "file not found";
#if CUDAPP_CUDA_VERSION >= 3010
          case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
            return "shared object symbol not found";
          case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
            return "shared object init failed";
#endif

          case CUDA_ERROR_INVALID_HANDLE: return "invalid handle";

          case CUDA_ERROR_NOT_FOUND: return "not found";

          case CUDA_ERROR_NOT_READY: return "not ready";

          case CUDA_ERROR_LAUNCH_FAILED: return "launch failed";
          case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: return "launch out of resources";
          case CUDA_ERROR_LAUNCH_TIMEOUT: return "launch timeout";
          case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: return "launch incompatible texturing";

#if CUDAPP_CUDA_VERSION >= 4000
          case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: return "peer access already enabled";
          case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED: return "peer access not enabled";
          case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE: return "primary context active";
          case CUDA_ERROR_CONTEXT_IS_DESTROYED: return "context is destroyed";
#endif

#if (CUDAPP_CUDA_VERSION >= 3000) && (CUDAPP_CUDA_VERSION < 3020)
          case CUDA_ERROR_POINTER_IS_64BIT:
             return "attempted to retrieve 64-bit pointer via 32-bit api function";
          case CUDA_ERROR_SIZE_IS_64BIT:
             return "attempted to retrieve 64-bit size via 32-bit api function";
#endif

#if CUDAPP_CUDA_VERSION >= 4010
          case CUDA_ERROR_ASSERT:
             return "device-side assert triggered";
          case CUDA_ERROR_TOO_MANY_PEERS:
             return "too many peers";
          case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
             return "host memory already registered";
          case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
             return "host memory not registered";
#endif

#if CUDAPP_CUDA_VERSION >= 5000
          case CUDA_ERROR_NOT_SUPPORTED:
             return "operation not supported on current system or device";
#endif

          case CUDA_ERROR_UNKNOWN: return "unknown";

          default: return "invalid/unknown error code";
        }
#endif
      }
  };

  struct cannot_activate_out_of_thread_context : public std::logic_error
  {
    cannot_activate_out_of_thread_context(std::string const &w)
      : std::logic_error(w)
    { }
  };

  struct cannot_activate_dead_context : public std::logic_error
  {
    cannot_activate_dead_context(std::string const &w)
      : std::logic_error(w)
    { }
  };

  // }}}

  // {{{ buffer interface helper

  class py_buffer_wrapper : public boost::noncopyable
  {
    private:
      bool m_initialized;

    public:
      Py_buffer m_buf;

      py_buffer_wrapper()
        : m_initialized(false)
      {}

      void get(PyObject *obj, int flags)
      {
        if (PyObject_GetBuffer(obj, &m_buf, flags))
          throw py::error_already_set();

        m_initialized = true;
      }

      virtual ~py_buffer_wrapper()
      {
        if (m_initialized)
          PyBuffer_Release(&m_buf);
      }
  };

  // }}}


  // {{{ version query ------------------------------------------------------------
#if CUDAPP_CUDA_VERSION >= 2020
  inline int get_driver_version()
  {
    int result;
    CUDAPP_CALL_GUARDED(cuDriverGetVersion, (&result));
    return result;
  }
#endif
  // }}}

  // {{{ device
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

#if CUDAPP_CUDA_VERSION >= 4010
      std::string pci_bus_id()
      {
        char buffer[1024];
        CUDAPP_CALL_GUARDED(cuDeviceGetPCIBusId, (buffer, sizeof(buffer), m_device));
        return buffer;
      }
#endif

      py::tuple compute_capability()
      {
        int major, minor;
        CUDAPP_CALL_GUARDED(cuDeviceComputeCapability, (&major, &minor, m_device));
        return py::make_tuple(major, minor);
      }

      pycuda_size_t total_memory()
      {
        pycuda_size_t bytes;

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

      hash_type hash() const
      {
        return m_device;
      }

      boost::shared_ptr<context> make_context(unsigned int flags);

      CUdevice handle() const
      { return m_device; }

#if CUDAPP_CUDA_VERSION >= 4000
      bool can_access_peer(device const &other)
      {
        int result;
        CUDAPP_CALL_GUARDED(cuDeviceCanAccessPeer, (&result, handle(), other.handle()));
        return result;
      }
#endif

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

#if CUDAPP_CUDA_VERSION >= 4010
  inline
  device *make_device_from_pci_bus_id(std::string const pci_bus_id)
  {
    CUdevice result;
    CUDAPP_CALL_GUARDED(cuDeviceGetByPCIBusId, (&result,
          const_cast<char *>(pci_bus_id.c_str())));
    return new device(result);
  }
#endif

  // }}}

  // {{{ context
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

  class context_stack;
  extern boost::thread_specific_ptr<context_stack> context_stack_ptr;

  class context_stack
  {
      /* This wrapper is necessary because we need to pop the contents
       * off the stack before we destroy each of the contexts. This, in turn,
       * is because the contexts need to be able to access the stack in order
       * to be destroyed.
       */
    private:
      typedef std::stack<boost::shared_ptr<context> > stack_t;
      typedef stack_t::value_type value_type;;
      stack_t m_stack;

    public:
      ~context_stack();

      bool empty() const
      { return m_stack.empty(); }

      value_type &top()
      { return m_stack.top(); }

      void pop()
      { m_stack.pop(); }

      void push(value_type v)
      { m_stack.push(v); }

      static context_stack &get()
      {
        if (context_stack_ptr.get() == 0)
          context_stack_ptr.reset(new context_stack);

        return *context_stack_ptr;
      }
  };

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
          /* It's possible that we get here with a non-zero m_use_count. Since the context
           * stack holds shared_ptrs, this must mean that the context stack itself is getting
           * destroyed, which means it's ok for this context to sign off, too.
           */
          detach();
        }
      }

      CUcontext handle() const
      { return m_context; }

      bool operator==(const context &other) const
      {
        return m_context == other.m_context;
      }

      bool operator!=(const context &other) const
      {
        return m_context != other.m_context;
      }

      hash_type hash() const
      {
        return hash_type(m_context) ^ hash_type(this);
      }

      boost::thread::id thread_id() const
      { return m_thread; }

      bool is_valid() const
      {
        return m_valid;
      }

      static boost::shared_ptr<context> attach(unsigned int flags)
      {
        CUcontext current;
        CUDAPP_CALL_GUARDED(cuCtxAttach, (&current, flags));
        boost::shared_ptr<context> result(new context(current));
        context_stack::get().push(result);
        return result;
      }

      void detach()
      {
        if (m_valid)
        {
          bool active_before_destruction = current_context().get() == this;
          if (active_before_destruction)
          {
            CUDAPP_CALL_GUARDED_CLEANUP(cuCtxDetach, (m_context));
          }
          else
          {
            if (m_thread == boost::this_thread::get_id())
            {
              CUDAPP_CALL_GUARDED_CLEANUP(cuCtxPushCurrent, (m_context));
              CUDAPP_CALL_GUARDED_CLEANUP(cuCtxDetach, (m_context));
              /* pop is implicit in detach */
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

          if (active_before_destruction)
          {
            boost::shared_ptr<context> new_active = current_context(this);
            if (new_active.get())
            {
              CUDAPP_CALL_GUARDED(cuCtxPushCurrent, (new_active->m_context));
            }
          }
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

#if CUDAPP_CUDA_VERSION >= 2000

      static void prepare_context_switch()
      {
        if (!context_stack::get().empty())
        {
          CUcontext popped;
          CUDAPP_CALL_GUARDED(cuCtxPopCurrent, (&popped));
        }
      }

      static void pop()
      {
        prepare_context_switch();
        context_stack &ctx_stack = context_stack::get();

        if (ctx_stack.empty())
        {
          throw error("context::pop", CUDA_ERROR_INVALID_CONTEXT,
              "cannot pop non-current context");
        }

        boost::shared_ptr<context> current = current_context();
        if (current)
          --current->m_use_count;

        ctx_stack.pop();

        current = current_context();
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
          if (context_stack::get().empty())
            return boost::shared_ptr<context>();

          boost::shared_ptr<context> result(context_stack::get().top());
          if (result.get() != except
              && result->is_valid())
          {
            // good, weak pointer didn't expire
            return result;
          }

          // context invalid, pop it and try again.
          context_stack::get().pop();
        }
      }

#if CUDAPP_CUDA_VERSION >= 3010
      static void set_limit(CUlimit limit, size_t value)
      {
        CUDAPP_CALL_GUARDED(cuCtxSetLimit, (limit, value));
      }

      static size_t get_limit(CUlimit limit)
      {
        size_t value;
        CUDAPP_CALL_GUARDED(cuCtxGetLimit, (&value, limit));
        return value;
      }
#endif

#if CUDAPP_CUDA_VERSION >= 3020
      static CUfunc_cache get_cache_config()
      {
        CUfunc_cache value;
        CUDAPP_CALL_GUARDED(cuCtxGetCacheConfig, (&value));
        return value;
      }

      static void set_cache_config(CUfunc_cache cc)
      {
        CUDAPP_CALL_GUARDED(cuCtxSetCacheConfig, (cc));
      }

      unsigned int get_api_version()
      {
        unsigned int value;
        CUDAPP_CALL_GUARDED(cuCtxGetApiVersion, (m_context, &value));
        return value;
      }
#endif

#if CUDAPP_CUDA_VERSION >= 4000
      static void enable_peer_access(context const &peer, unsigned int flags)
      {
        CUDAPP_CALL_GUARDED(cuCtxEnablePeerAccess, (peer.handle(), flags));
      }

      static void disable_peer_access(context const &peer)
      {
        CUDAPP_CALL_GUARDED(cuCtxDisablePeerAccess, (peer.handle()));
      }
#endif

#if CUDAPP_CUDA_VERSION >= 4020
      static CUsharedconfig get_shared_config()
      {
        CUsharedconfig config;
        CUDAPP_CALL_GUARDED(cuCtxGetSharedMemConfig, (&config));
        return config;
      }

      static void set_shared_config(CUsharedconfig config)
      {
        CUDAPP_CALL_GUARDED(cuCtxSetSharedMemConfig, (config));
      }
#endif

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
    context_stack::get().push(result);
    return result;
  }








#if CUDAPP_CUDA_VERSION >= 2000
  inline
  void context_push(boost::shared_ptr<context> ctx)
  {
    context::prepare_context_switch();

    CUDAPP_CALL_GUARDED(cuCtxPushCurrent, (ctx->m_context));
    context_stack::get().push(ctx);
    ++ctx->m_use_count;
  }
#endif




  inline context_stack::~context_stack()
  {
    if (!m_stack.empty())
    {
      std::cerr
        << "-------------------------------------------------------------------" << std::endl
        << "PyCUDA ERROR: The context stack was not empty upon module cleanup." << std::endl
        << "-------------------------------------------------------------------" << std::endl
        << "A context was still active when the context stack was being" << std::endl
        << "cleaned up. At this point in our execution, CUDA may already" << std::endl
        << "have been deinitialized, so there is no way we can finish" << std::endl
        << "cleanly. The program will be aborted now." << std::endl
        << "Use Context.pop() to avoid this problem." << std::endl
        << "-------------------------------------------------------------------" << std::endl;
      abort();
    }
  }




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
        if (!m_context->is_valid())
          throw pycuda::cannot_activate_dead_context(
              "cannot activate dead context");

        m_did_switch = context::current_context() != m_context;
        if (m_did_switch)
        {
          if (boost::this_thread::get_id() != m_context->thread_id())
            throw pycuda::cannot_activate_out_of_thread_context(
                "cannot activate out-of-thread context");
#if CUDAPP_CUDA_VERSION >= 2000
          context_push(m_context);
#else
          throw pycuda::error("scoped_context_activation", CUDA_ERROR_INVALID_CONTEXT,
              "not available in CUDA < 2.0");
#endif
        }
      }

      ~scoped_context_activation()
      {
#if CUDAPP_CUDA_VERSION >= 2000
        if (m_did_switch)
          m_context->pop();
#endif
      }

  };

  // }}}

  // {{{ stream
  class event;

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
          CUDAPP_CALL_GUARDED_CLEANUP(cuStreamDestroy, (m_stream));
        }
        CUDAPP_CATCH_CLEANUP_ON_DEAD_CONTEXT(stream);
      }

      void synchronize()
      { CUDAPP_CALL_GUARDED_THREADED(cuStreamSynchronize, (m_stream)); }

      CUstream handle() const
      { return m_stream; }

      intptr_t handle_int() const
      { return (intptr_t) m_stream; }

#if CUDAPP_CUDA_VERSION >= 3020
      void wait_for_event(const event &evt);
#endif

      bool is_done() const
      {
        CUDAPP_PRINT_CALL_TRACE("cuStreamQuery");
        CUresult result = cuStreamQuery(m_stream);
        switch (result)
        {
          case CUDA_SUCCESS:
            return true;
          case CUDA_ERROR_NOT_READY:
            return false;
          default:
            CUDAPP_PRINT_ERROR_TRACE("cuStreamQuery", result);
            throw error("cuStreamQuery", result);
        }
      }
  };

  // }}}

  // {{{ array
  class array : public boost::noncopyable, public context_dependent
  {
    private:
      CUarray m_array;
      bool m_managed;

    public:
      array(const CUDA_ARRAY_DESCRIPTOR &descr)
        : m_managed(true)
      { CUDAPP_CALL_GUARDED(cuArrayCreate, (&m_array, &descr)); }

#if CUDAPP_CUDA_VERSION >= 2000
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
            CUDAPP_CALL_GUARDED_CLEANUP(cuArrayDestroy, (m_array));
          }
          CUDAPP_CATCH_CLEANUP_ON_DEAD_CONTEXT(array);

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

#if CUDAPP_CUDA_VERSION >= 2000
      CUDA_ARRAY3D_DESCRIPTOR get_descriptor_3d()
      {
        CUDA_ARRAY3D_DESCRIPTOR result;
        CUDAPP_CALL_GUARDED(cuArray3DGetDescriptor, (&result, m_array));
        return result;
      }
#endif

      CUarray handle() const
      { return m_array; }

    intptr_t handle_int() const
    { return  (intptr_t) m_array; }
  };

  // }}}

  // {{{ texture reference
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
          CUDAPP_CALL_GUARDED_CLEANUP(cuTexRefDestroy, (m_texref));
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

      pycuda_size_t set_address(CUdeviceptr dptr, unsigned int bytes, bool allow_offset=false)
      {
        pycuda_size_t byte_offset;
        CUDAPP_CALL_GUARDED(cuTexRefSetAddress, (&byte_offset,
              m_texref, dptr, bytes));

        if (!allow_offset && byte_offset != 0)
          throw pycuda::error("texture_reference::set_address", CUDA_ERROR_INVALID_VALUE,
              "texture binding resulted in offset, but allow_offset was false");

        m_array.reset();
        return byte_offset;
      }

#if CUDAPP_CUDA_VERSION >= 2020
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

#if CUDAPP_CUDA_VERSION >= 2000
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

  // }}}

  // {{{ surface reference
#if CUDAPP_CUDA_VERSION >= 3010
  class module;

  class surface_reference : public  boost::noncopyable
  {
    private:
      CUsurfref m_surfref;

      // life support for array and module
      boost::shared_ptr<array> m_array;
      boost::shared_ptr<module> m_module;

    public:
      surface_reference(CUsurfref sr)
        : m_surfref(sr)
      { }

      void set_module(boost::shared_ptr<module> mod)
      { m_module = mod; }

      CUsurfref handle() const
      { return m_surfref; }

      void set_array(boost::shared_ptr<array> ary, unsigned int flags)
      {
        CUDAPP_CALL_GUARDED(cuSurfRefSetArray, (m_surfref, ary->handle(), flags));
        m_array = ary;
      }

      array *get_array()
      {
        CUarray result;
        CUDAPP_CALL_GUARDED(cuSurfRefGetArray, (&result, m_surfref));
        return new array(result, false);
      }
  };
#endif

  // }}}

  // {{{ module
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
          CUDAPP_CALL_GUARDED_CLEANUP(cuModuleUnload, (m_module));
        }
        CUDAPP_CATCH_CLEANUP_ON_DEAD_CONTEXT(module);
      }

      CUmodule handle() const
      { return m_module; }

      function get_function(const char *name);
      py::tuple get_global(const char *name)
      {
        CUdeviceptr devptr;
        pycuda_size_t bytes;
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
  texture_reference *module_get_texref(
      boost::shared_ptr<module> mod, const char *name)
  {
    CUtexref tr;
    CUDAPP_CALL_GUARDED(cuModuleGetTexRef, (&tr, mod->handle(), name));
    std::auto_ptr<texture_reference> result(
        new texture_reference(tr, false));
    result->set_module(mod);
    return result.release();
  }

#if CUDAPP_CUDA_VERSION >= 3010
  inline
  surface_reference *module_get_surfref(
      boost::shared_ptr<module> mod, const char *name)
  {
    CUsurfref sr;
    CUDAPP_CALL_GUARDED(cuModuleGetSurfRef, (&sr, mod->handle(), name));
    std::auto_ptr<surface_reference> result(
        new surface_reference(sr));
    result->set_module(mod);
    return result.release();
  }
#endif

  // }}}

  // {{{ function
  class function
  {
    private:
      CUfunction m_function;
      std::string m_symbol;

    public:
      function(CUfunction func, std::string const &sym)
        : m_function(func), m_symbol(sym)
      { }

      void set_block_shape(int x, int y, int z)
      {
        CUDAPP_CALL_GUARDED_WITH_TRACE_INFO(
            cuFuncSetBlockShape, (m_function, x, y, z), m_symbol);
      }
      void set_shared_size(unsigned int bytes)
      {
        CUDAPP_CALL_GUARDED_WITH_TRACE_INFO(
            cuFuncSetSharedSize, (m_function, bytes), m_symbol);
      }

      void param_set_size(unsigned int bytes)
      {
        CUDAPP_CALL_GUARDED_WITH_TRACE_INFO(
            cuParamSetSize, (m_function, bytes), m_symbol);
      }
      void param_set(int offset, unsigned int value)
      {
        CUDAPP_CALL_GUARDED_WITH_TRACE_INFO(
            cuParamSeti, (m_function, offset, value), m_symbol);
      }
      void param_set(int offset, float value)
      {
        CUDAPP_CALL_GUARDED_WITH_TRACE_INFO(
          cuParamSetf, (m_function, offset, value), m_symbol);
      }
      void param_setv(int offset, void *buf, size_t len)
      {
        // maybe the unsigned int will change, it does not seem right
        CUDAPP_CALL_GUARDED_WITH_TRACE_INFO(
          cuParamSetv, (m_function, offset, buf, (unsigned int) len), m_symbol);
      }
      void param_set_texref(const texture_reference &tr)
      {
        CUDAPP_CALL_GUARDED_WITH_TRACE_INFO(cuParamSetTexRef, (m_function,
            CU_PARAM_TR_DEFAULT, tr.handle()), m_symbol);
      }

      void launch()
      {
        CUDAPP_CALL_GUARDED_THREADED_WITH_TRACE_INFO(
            cuLaunch, (m_function), m_symbol);
      }
      void launch_grid(int grid_width, int grid_height)
      {
        CUDAPP_CALL_GUARDED_THREADED_WITH_TRACE_INFO(
          cuLaunchGrid, (m_function, grid_width, grid_height), m_symbol);
      }
      void launch_grid_async(int grid_width, int grid_height, const stream &s)
      {
        CUDAPP_CALL_GUARDED_THREADED_WITH_TRACE_INFO(
            cuLaunchGridAsync, (m_function, grid_width, grid_height, s.handle()),
            m_symbol);
      }

#if CUDAPP_CUDA_VERSION >= 2020
      int get_attribute(CUfunction_attribute attr) const
      {
        int result;
        CUDAPP_CALL_GUARDED_WITH_TRACE_INFO(
            cuFuncGetAttribute, (&result, attr, m_function), m_symbol);
        return result;
      }
#endif

#if CUDAPP_CUDA_VERSION >= 3000 && defined(CUDAPP_POST_30_BETA)
      void set_cache_config(CUfunc_cache fc)
      {
        CUDAPP_CALL_GUARDED_WITH_TRACE_INFO(
            cuFuncSetCacheConfig, (m_function, fc), m_symbol);
      }
#endif

#if CUDAPP_CUDA_VERSION >= 4000
      void launch_kernel(py::tuple grid_dim_py, py::tuple block_dim_py,
          py::object parameter_buffer,
          unsigned shared_mem_bytes, py::object stream_py)
      {
        const unsigned axis_count = 3;
        unsigned grid_dim[axis_count];
        unsigned block_dim[axis_count];

        for (unsigned i = 0; i < axis_count; ++i)
        {
          grid_dim[i] = 1;
          block_dim[i] = 1;
        }

        pycuda_size_t gd_length = py::len(grid_dim_py);
        if (gd_length > axis_count)
          throw pycuda::error("function::launch_kernel", CUDA_ERROR_INVALID_HANDLE,
              "too many grid dimensions in kernel launch");

        for (unsigned i = 0; i < gd_length; ++i)
          grid_dim[i] = py::extract<unsigned>(grid_dim_py[i]);

        pycuda_size_t bd_length = py::len(block_dim_py);
        if (bd_length > axis_count)
          throw pycuda::error("function::launch_kernel", CUDA_ERROR_INVALID_HANDLE,
              "too many block dimensions in kernel launch");

        for (unsigned i = 0; i < bd_length; ++i)
          block_dim[i] = py::extract<unsigned>(block_dim_py[i]);

        PYCUDA_PARSE_STREAM_PY;

        py_buffer_wrapper par_buf_wrapper;
        par_buf_wrapper.get(parameter_buffer.ptr(), PyBUF_ANY_CONTIGUOUS);
        size_t par_len = par_buf_wrapper.m_buf.len;

        void *config[] = {
          CU_LAUNCH_PARAM_BUFFER_POINTER, const_cast<void *>(par_buf_wrapper.m_buf.buf),
          CU_LAUNCH_PARAM_BUFFER_SIZE, &par_len,
          CU_LAUNCH_PARAM_END
        };

        CUDAPP_CALL_GUARDED(
            cuLaunchKernel, (m_function,
              grid_dim[0], grid_dim[1], grid_dim[2],
              block_dim[0], block_dim[1], block_dim[2],
              shared_mem_bytes, s_handle, 0, config
              ));
      }

#endif

#if CUDAPP_CUDA_VERSION >= 4020
      void set_shared_config(CUsharedconfig config)
      {
        CUDAPP_CALL_GUARDED_WITH_TRACE_INFO(
            cuFuncSetSharedMemConfig, (m_function, config), m_symbol);
      }
#endif

  };

  inline
  function module::get_function(const char *name)
  {
    CUfunction func;
    CUDAPP_CALL_GUARDED(cuModuleGetFunction, (&func, m_module, name));
    return function(func, name);
  }

  // }}}

  // {{{ device memory
  inline
  py::tuple mem_get_info()
  {
    pycuda_size_t free, total;
    CUDAPP_CALL_GUARDED(cuMemGetInfo, (&free, &total));
    return py::make_tuple(free, total);
  }

  inline
  CUdeviceptr mem_alloc(size_t bytes)
  {
    CUdeviceptr devptr;
    CUDAPP_CALL_GUARDED(cuMemAlloc, (&devptr, bytes));
    return devptr;
  }

  inline
  void mem_free(CUdeviceptr devptr)
  {
    CUDAPP_CALL_GUARDED_CLEANUP(cuMemFree, (devptr));
  }

  // A class the user can override to make device_allocation-
  // workalikes.

  class pointer_holder_base
  {
    public:
      virtual ~pointer_holder_base() { }
      virtual CUdeviceptr get_pointer() = 0;

      operator CUdeviceptr()
      { return get_pointer(); }

      py::object as_buffer(size_t size, size_t offset)
      {
        return py::object(
            py::handle<>(
#if PY_VERSION_HEX >= 0x03030000
              PyMemoryView_FromMemory((char *) (get_pointer() + offset), size,
                PyBUF_WRITE)
#else /* Py2 */
              PyBuffer_FromReadWriteMemory((void *) (get_pointer() + offset), size)
#endif
              ));
      }
  };

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
          CUDAPP_CATCH_CLEANUP_ON_DEAD_CONTEXT(device_allocation);

          release_context();
          m_valid = false;
        }
        else
          throw pycuda::error("device_allocation::free", CUDA_ERROR_INVALID_HANDLE);
      }

      ~device_allocation()
      {
        if (m_valid)
          free();
      }

      operator CUdeviceptr() const
      { return m_devptr; }

      py::object as_buffer(size_t size, size_t offset)
      {
        return py::object(
            py::handle<>(
#if PY_VERSION_HEX >= 0x03030000
              PyMemoryView_FromMemory((char *) (m_devptr + offset), size,
                PyBUF_READ | PyBUF_WRITE)
#else /* Py2 */
              PyBuffer_FromReadWriteMemory((void *) (m_devptr + offset), size)
#endif
              ));
      }
  };

  inline Py_ssize_t mem_alloc_pitch(
      std::auto_ptr<device_allocation> &da,
        unsigned int width, unsigned int height, unsigned int access_size)
  {
    CUdeviceptr devptr;
    pycuda_size_t pitch;
    CUDAPP_CALL_GUARDED(cuMemAllocPitch, (&devptr, &pitch, width, height, access_size));
    da = std::auto_ptr<device_allocation>(new device_allocation(devptr));
    return pitch;
  }

  inline
  py::tuple mem_get_address_range(CUdeviceptr ptr)
  {
    CUdeviceptr base;
    pycuda_size_t size;
    CUDAPP_CALL_GUARDED(cuMemGetAddressRange, (&base, &size, ptr));
    return py::make_tuple(base, size);
  }

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

  // }}}

  // {{{ ipc_mem_handle

#if CUDAPP_CUDA_VERSION >= 4010 && PY_VERSION_HEX >= 0x02060000
  class ipc_mem_handle : public boost::noncopyable, public context_dependent
  {
    private:
      bool m_valid;

    protected:
      CUdeviceptr m_devptr;

    public:
      ipc_mem_handle(py::object obj, CUipcMem_flags flags=CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS)
        : m_valid(true)
      {
        if (!PyByteArray_Check(obj.ptr()))
          throw pycuda::error("event_from_ipc_handle", CUDA_ERROR_INVALID_VALUE,
              "argument is not a bytes array");
        CUipcMemHandle handle;
        if (PyByteArray_GET_SIZE(obj.ptr()) != sizeof(handle))
          throw pycuda::error("event_from_ipc_handle", CUDA_ERROR_INVALID_VALUE,
              "handle has the wrong size");
        memcpy(&handle, PyByteArray_AS_STRING(obj.ptr()), sizeof(handle));

        CUDAPP_CALL_GUARDED(cuIpcOpenMemHandle, (&m_devptr, handle, flags));
      }

      void close()
      {
        if (m_valid)
        {
          try
          {
            scoped_context_activation ca(get_context());
            CUDAPP_CALL_GUARDED_CLEANUP(cuIpcCloseMemHandle, (m_devptr));
          }
          CUDAPP_CATCH_CLEANUP_ON_DEAD_CONTEXT(ipc_mem_handle);

          release_context();
          m_valid = false;
        }
        else
          throw pycuda::error("ipc_mem_handle::close", CUDA_ERROR_INVALID_HANDLE);
      }

      ~ipc_mem_handle()
      {
        if (m_valid)
          close();
      }

      operator CUdeviceptr() const
      { return m_devptr; }
  };

  inline
  py::object mem_get_ipc_handle(CUdeviceptr devptr)
  {
    CUipcMemHandle handle;
    CUDAPP_CALL_GUARDED(cuIpcGetMemHandle, (&handle, devptr));
    return py::object(py::handle<>(PyByteArray_FromStringAndSize(
            reinterpret_cast<const char *>(&handle),
            sizeof(handle))));
  }

#endif

  // }}}

  // {{{ structured memcpy

#define MEMCPY_SETTERS \
    void set_src_host(py::object buf_py) \
    { \
      srcMemoryType = CU_MEMORYTYPE_HOST; \
      py_buffer_wrapper buf_wrapper; \
      buf_wrapper.get(buf_py.ptr(), PyBUF_STRIDED_RO); \
      srcHost = buf_wrapper.m_buf.buf; \
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
      py_buffer_wrapper buf_wrapper; \
      buf_wrapper.get(buf_py.ptr(), PyBUF_STRIDED); \
      dstHost = buf_wrapper.m_buf.buf; \
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

#if CUDAPP_CUDA_VERSION >= 4000
#define MEMCPY_SETTERS_UNIFIED \
    void set_src_unified(py::object buf_py) \
    { \
      srcMemoryType = CU_MEMORYTYPE_UNIFIED; \
      py_buffer_wrapper buf_wrapper; \
      buf_wrapper.get(buf_py.ptr(), PyBUF_ANY_CONTIGUOUS); \
      srcHost = buf_wrapper.m_buf.buf; \
    } \
    \
    void set_dst_unified(py::object buf_py) \
    { \
      dstMemoryType = CU_MEMORYTYPE_UNIFIED; \
      py_buffer_wrapper buf_wrapper; \
      buf_wrapper.get(buf_py.ptr(), PyBUF_ANY_CONTIGUOUS | PyBUF_WRITABLE); \
      dstHost = buf_wrapper.m_buf.buf; \
    }
#else
#define MEMCPY_SETTERS_UNIFIED /* empty */
#endif





  struct memcpy_2d : public CUDA_MEMCPY2D
  {
    memcpy_2d() : CUDA_MEMCPY2D()
    {
    }

    MEMCPY_SETTERS;
    MEMCPY_SETTERS_UNIFIED;

    void execute(bool aligned=false) const
    {
      if (aligned)
      { CUDAPP_CALL_GUARDED_THREADED(cuMemcpy2D, (this)); }
      else
      { CUDAPP_CALL_GUARDED_THREADED(cuMemcpy2DUnaligned, (this)); }
    }

    void execute_async(const stream &s) const
    { CUDAPP_CALL_GUARDED_THREADED(cuMemcpy2DAsync, (this, s.handle())); }
  };

#if CUDAPP_CUDA_VERSION >= 2000
  struct memcpy_3d : public CUDA_MEMCPY3D
  {
    memcpy_3d() : CUDA_MEMCPY3D()
    {
    }

    MEMCPY_SETTERS;
    MEMCPY_SETTERS_UNIFIED;

    void execute() const
    {
      CUDAPP_CALL_GUARDED_THREADED(cuMemcpy3D, (this));
    }

    void execute_async(const stream &s) const
    { CUDAPP_CALL_GUARDED_THREADED(cuMemcpy3DAsync, (this, s.handle())); }
  };
#endif

#if CUDAPP_CUDA_VERSION >= 4000
  struct memcpy_3d_peer : public CUDA_MEMCPY3D_PEER
  {
    memcpy_3d_peer() : CUDA_MEMCPY3D_PEER()
    {
    }

    MEMCPY_SETTERS;
    MEMCPY_SETTERS_UNIFIED;

    void set_src_context(context const &ctx)
    {
      srcContext = ctx.handle();
    }

    void set_dst_context(context const &ctx)
    {
      dstContext = ctx.handle();
    }

    void execute() const
    {
      CUDAPP_CALL_GUARDED_THREADED(cuMemcpy3DPeer, (this));
    }

    void execute_async(const stream &s) const
    { CUDAPP_CALL_GUARDED_THREADED(cuMemcpy3DPeerAsync, (this, s.handle())); }
  };
#endif

  // }}}

  // {{{ host memory
  inline void *mem_host_alloc(size_t size, unsigned flags=0)
  {
    void *m_data;
#if CUDAPP_CUDA_VERSION >= 2020
    CUDAPP_CALL_GUARDED(cuMemHostAlloc, (&m_data, size, flags));
#else
    if (flags != 0)
      throw pycuda::error("mem_host_alloc", CUDA_ERROR_INVALID_VALUE,
          "nonzero flags in mem_host_alloc not allowed in CUDA 2.1 and older");
    CUDAPP_CALL_GUARDED(cuMemAllocHost, (&m_data, size));
#endif
    return m_data;
  }

  inline void mem_host_free(void *ptr)
  {
    CUDAPP_CALL_GUARDED_CLEANUP(cuMemFreeHost, (ptr));
  }

#if CUDAPP_CUDA_VERSION >= 6000
  inline CUdeviceptr mem_managed_alloc(size_t size, unsigned flags=0)
  {
    CUdeviceptr m_data;
    CUDAPP_CALL_GUARDED(cuMemAllocManaged, (&m_data, size, flags));
    return m_data;
  }
#endif

#if CUDAPP_CUDA_VERSION >= 4000
  inline void *mem_host_register(void *ptr, size_t bytes, unsigned int flags=0)
  {
    CUDAPP_CALL_GUARDED(cuMemHostRegister, (ptr, bytes, flags));
    return ptr;
  }

  inline void mem_host_unregister(void *ptr)
  {
    CUDAPP_CALL_GUARDED_CLEANUP(cuMemHostUnregister, (ptr));
  }
#endif

  inline void *aligned_malloc(size_t size, size_t alignment, void **original_pointer)
  {
    // alignment must be a power of two.
    if ((alignment & (alignment - 1)) != 0)
      throw pycuda::error("aligned_malloc", CUDA_ERROR_INVALID_VALUE,
          "alignment must be a power of two");

    if (alignment == 0)
      throw pycuda::error("aligned_malloc", CUDA_ERROR_INVALID_VALUE,
          "alignment must non-zero");

    void *p = malloc(size + (alignment - 1));
    if (!p)
      throw pycuda::error("aligned_malloc", CUDA_ERROR_OUT_OF_MEMORY,
          "aligned malloc failed");

    *original_pointer = p;

    p = (void *)((((ptrdiff_t)(p)) + (alignment-1)) & -alignment);
    return p;
  }



  struct host_pointer : public boost::noncopyable, public context_dependent
  {
    protected:
      bool m_valid;
      void *m_data;

    public:
      host_pointer()
        : m_valid(false)
      { }

      host_pointer(void *ptr)
        : m_valid(true), m_data(ptr)
      { }

      virtual ~host_pointer()
      { }

      void *data()
      { return m_data; }

#if CUDAPP_CUDA_VERSION >= 2020
      CUdeviceptr get_device_pointer()
      {
        CUdeviceptr result;
        CUDAPP_CALL_GUARDED(cuMemHostGetDevicePointer, (&result, m_data, 0));
        return result;
      }
#endif

  };

  struct pagelocked_host_allocation : public host_pointer
  {
    public:
      pagelocked_host_allocation(size_t bytesize, unsigned flags=0)
        : host_pointer(mem_host_alloc(bytesize, flags))
      { }

      /* Don't try to be clever and coalesce these in the base class.
       * Won't work: Destructors may not call virtual functions.
       */
      ~pagelocked_host_allocation()
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
            mem_host_free(m_data);
          }
          CUDAPP_CATCH_CLEANUP_ON_DEAD_CONTEXT(pagelocked_host_allocation);

          release_context();
          m_valid = false;
        }
        else
          throw pycuda::error("pagelocked_host_allocation::free", CUDA_ERROR_INVALID_HANDLE);
      }

#if CUDAPP_CUDA_VERSION >= 3020
      unsigned int get_flags()
      {
        unsigned int flags;
        CUDAPP_CALL_GUARDED(cuMemHostGetFlags, (&flags, m_data));
        return flags;
      }
#endif
  };

  struct aligned_host_allocation : public host_pointer
  {
      void *m_original_pointer;

    public:
      aligned_host_allocation(size_t size, size_t alignment)
        : host_pointer(aligned_malloc(size, alignment, &m_original_pointer))
      { }

      /* Don't try to be clever and coalesce these in the base class.
       * Won't work: Destructors may not call virtual functions.
       */
      ~aligned_host_allocation()
      {
        if (m_valid)
          free();
      }

      void free()
      {
        if (m_valid)
        {
          ::free(m_original_pointer);
          m_valid = false;
        }
        else
          throw pycuda::error("aligned_host_allocation::free", CUDA_ERROR_INVALID_HANDLE);
      }
  };

#if CUDAPP_CUDA_VERSION >= 6000
  struct managed_allocation : public device_allocation
  {
    public:
      managed_allocation(size_t bytesize, unsigned flags=0)
        : device_allocation(mem_managed_alloc(bytesize, flags))
      { }

      // The device pointer is also valid on the host
      void *data()
      { return (void *) m_devptr; }

      CUdeviceptr get_device_pointer()
      {
        return m_devptr;
      }

      void attach(unsigned flags, py::object stream_py)
      {
        PYCUDA_PARSE_STREAM_PY;

        CUDAPP_CALL_GUARDED(cuStreamAttachMemAsync, (s_handle, m_devptr, 0, flags));
      }

  };
#endif


#if CUDAPP_CUDA_VERSION >= 4000
  struct registered_host_memory : public host_pointer
  {
    private:
      py::object m_base;

    public:
      registered_host_memory(void *p, size_t bytes, unsigned int flags=0,
          py::object base=py::object())
        : host_pointer(mem_host_register(p, bytes, flags)), m_base(base)
      {
      }

      /* Don't try to be clever and coalesce these in the base class.
       * Won't work: Destructors may not call virtual functions.
       */
      ~registered_host_memory()
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
            mem_host_unregister(m_data);
          }
          CUDAPP_CATCH_CLEANUP_ON_DEAD_CONTEXT(host_allocation);

          release_context();
          m_valid = false;
        }
        else
          throw pycuda::error("registered_host_memory::free", CUDA_ERROR_INVALID_HANDLE);
      }

      py::object base() const
      {
        return m_base;
      }
  };
#endif

  // }}}

  // {{{ event
  class event : public boost::noncopyable, public context_dependent
  {
    private:
      CUevent m_event;

    public:
      event(unsigned int flags=0)
      { CUDAPP_CALL_GUARDED(cuEventCreate, (&m_event, flags)); }

      event(CUevent evt)
      : m_event(evt)
      { }

      ~event()
      {
        try
        {
          scoped_context_activation ca(get_context());
          CUDAPP_CALL_GUARDED_CLEANUP(cuEventDestroy, (m_event));
        }
        CUDAPP_CATCH_CLEANUP_ON_DEAD_CONTEXT(event);
      }

      event *record(py::object stream_py)
      {
        PYCUDA_PARSE_STREAM_PY;

        CUDAPP_CALL_GUARDED(cuEventRecord, (m_event, s_handle));
        return this;
      }

      CUevent handle() const
      { return m_event; }

      event *synchronize()
      {
        CUDAPP_CALL_GUARDED_THREADED(cuEventSynchronize, (m_event));
        return this;
      }

      bool query() const
      {
        CUDAPP_PRINT_CALL_TRACE("cuEventQuery");

        CUresult result = cuEventQuery(m_event);
        switch (result)
        {
          case CUDA_SUCCESS:
            return true;
          case CUDA_ERROR_NOT_READY:
            return false;
          default:
            CUDAPP_PRINT_ERROR_TRACE("cuEventQuery", result);
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

#if CUDAPP_CUDA_VERSION >= 4010 && PY_VERSION_HEX >= 0x02060000
      py::object ipc_handle()
      {
        CUipcEventHandle handle;
        CUDAPP_CALL_GUARDED(cuIpcGetEventHandle, (&handle, m_event));
        return py::object(py::handle<>(PyByteArray_FromStringAndSize(
              reinterpret_cast<const char *>(&handle),
              sizeof(handle))));
      }
#endif
  };

#if CUDAPP_CUDA_VERSION >= 3020
  inline void stream::wait_for_event(const event &evt)
  {
    CUDAPP_CALL_GUARDED(cuStreamWaitEvent, (m_stream, evt.handle(), 0));
  }
#endif

#if CUDAPP_CUDA_VERSION >= 4010 && PY_VERSION_HEX >= 0x02060000
  inline
  event *event_from_ipc_handle(py::object obj)
  {
    if (!PyByteArray_Check(obj.ptr()))
      throw pycuda::error("event_from_ipc_handle", CUDA_ERROR_INVALID_VALUE,
          "argument is not a bytes array");
    CUipcEventHandle handle;
    if (PyByteArray_GET_SIZE(obj.ptr()) != sizeof(handle))
      throw pycuda::error("event_from_ipc_handle", CUDA_ERROR_INVALID_VALUE,
          "handle has the wrong size");
    memcpy(&handle, PyByteArray_AS_STRING(obj.ptr()), sizeof(handle));

    CUevent evt;
    CUDAPP_CALL_GUARDED(cuIpcOpenEventHandle, (&evt, handle));

    return new event(evt);
  }
#endif

  // }}}

  // {{{ profiler
#if CUDAPP_CUDA_VERSION >= 4000
  inline void initialize_profiler(
      const char *config_file,
      const char *output_file,
      CUoutput_mode output_mode)
  {
    CUDAPP_CALL_GUARDED(cuProfilerInitialize, (config_file, output_file, output_mode));
  }

  inline void start_profiler()
  {
    CUDAPP_CALL_GUARDED(cuProfilerStart, ());
  }

  inline void stop_profiler()
  {
    CUDAPP_CALL_GUARDED(cuProfilerStop, ());
  }
#endif
  // }}}
}




#endif
// vim: foldmethod=marker
