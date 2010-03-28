#include <cuda.hpp>

#include <utility>
#include <numeric>
#include <algorithm>

#include "tools.hpp"
#include "wrap_helpers.hpp"
#include <boost/python/stl_iterator.hpp>




#if CUDA_VERSION < 1010
#error PyCuda only works with CUDA 1.1 or newer.
#endif




using namespace cuda;
using boost::shared_ptr;




namespace
{
  py::handle<> 
    CudaError, 
    CudaMemoryError, 
    CudaLogicError, 
    CudaRuntimeError,
    CudaLaunchError;




  void translate_cuda_error(const cuda::error &err)
  {
    if (err.code() == CUDA_ERROR_LAUNCH_FAILED
        || err.code() == CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
        || err.code() == CUDA_ERROR_LAUNCH_TIMEOUT
        || err.code() == CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING
       )
      PyErr_SetString(CudaLaunchError.get(), err.what());
    else if (err.code() == CUDA_ERROR_OUT_OF_MEMORY)
      PyErr_SetString(CudaMemoryError.get(), err.what());
    else if (err.code() == CUDA_ERROR_NO_DEVICE
        || err.code() == CUDA_ERROR_NO_BINARY_FOR_GPU
        || err.code() == CUDA_ERROR_NO_BINARY_FOR_GPU
        || err.code() == CUDA_ERROR_FILE_NOT_FOUND
        || err.code() == CUDA_ERROR_NOT_READY
#if CUDA_VERSION >= 3000 && defined(CUDAPP_POST_30_BETA)
        || err.code() == CUDA_ERROR_ECC_UNCORRECTABLE
#endif
        )
      PyErr_SetString(CudaRuntimeError.get(), err.what());
    else if (err.code() == CUDA_ERROR_UNKNOWN)
      PyErr_SetString(CudaError.get(), err.what());
    else 
      PyErr_SetString(CudaLogicError.get(), err.what());
  }




  py::tuple cuda_version()
  {
    return py::make_tuple(
        CUDA_VERSION / 1000, 
        (CUDA_VERSION % 1000)/10, 
        CUDA_VERSION % 10);
  }



  class host_alloc_flags { };
  class array3d_flags { };



  py::object device_get_attribute(device const &dev, CUdevice_attribute attr)
  {
#if CUDA_VERSION >= 2020
    if (attr == CU_DEVICE_ATTRIBUTE_COMPUTE_MODE)
      return py::object(CUcomputemode(dev.get_attribute(attr)));
    else
#endif
      return py::object(dev.get_attribute(attr));
  }



  device_allocation *mem_alloc_wrap(unsigned long bytes)
  {
    return new device_allocation(pycuda::mem_alloc_gc(bytes));
  }

  py::tuple mem_alloc_pitch_wrap(
      unsigned int width, unsigned int height, unsigned int access_size)
  {
    std::auto_ptr<device_allocation> da;
    unsigned int pitch = mem_alloc_pitch(
        da, width, height, access_size);
    return py::make_tuple(
        handle_from_new_ptr(da.release()), pitch);
  }




  void  py_memset_d8(CUdeviceptr dst, unsigned char uc, unsigned int n ) 
  { CUDAPP_CALL_GUARDED_THREADED(cuMemsetD8, (dst, uc, n )); }
  void  py_memset_d16(CUdeviceptr dst, unsigned short us, unsigned int n ) 
  { CUDAPP_CALL_GUARDED_THREADED(cuMemsetD16, (dst, us, n )); }
  void  py_memset_d32(CUdeviceptr dst, unsigned int ui, unsigned int n ) 
  { CUDAPP_CALL_GUARDED_THREADED(cuMemsetD32, (dst, ui, n )); }

  void  py_memset_d2d8(CUdeviceptr dst, unsigned int dst_pitch, 
      unsigned char uc, unsigned int width, unsigned int height )
  { CUDAPP_CALL_GUARDED_THREADED(cuMemsetD2D8, (dst, dst_pitch, uc, width, height)); }

  void  py_memset_d2d16(CUdeviceptr dst, unsigned int dst_pitch, 
      unsigned short us, unsigned int width, unsigned int height )
  { CUDAPP_CALL_GUARDED_THREADED(cuMemsetD2D16, (dst, dst_pitch, us, width, height)); }

  void  py_memset_d2d32(CUdeviceptr dst, unsigned int dst_pitch, 
      unsigned int ui, unsigned int width, unsigned int height )
  { CUDAPP_CALL_GUARDED_THREADED(cuMemsetD2D32, (dst, dst_pitch, ui, width, height)); }

  void  py_memcpy_dtod(CUdeviceptr dst, CUdeviceptr src, 
      unsigned int byte_count)
  { CUDAPP_CALL_GUARDED_THREADED(cuMemcpyDtoD, (dst, src, byte_count)); }




#if CUDA_VERSION >= 3000
  void  py_memcpy_dtod_async(CUdeviceptr dst, CUdeviceptr src, 
      unsigned int byte_count, py::object stream_py)
  {
    CUstream s_handle;
    if (stream_py.ptr() != Py_None)
    {
      const stream &s = py::extract<const stream &>(stream_py);
      s_handle = s.handle();
    }
    else
      s_handle = 0;

    CUDAPP_CALL_GUARDED_THREADED(cuMemcpyDtoDAsync, 
        (dst, src, byte_count, s_handle)); 
  }
#endif




  void function_param_setv(function &f, int offset, py::object buffer)
  { 
    const void *buf;
    PYCUDA_BUFFER_SIZE_T len;
    if (PyObject_AsReadBuffer(buffer.ptr(), &buf, &len))
      throw py::error_already_set();
    f.param_setv(offset, const_cast<void *>(buf), len);
  }




  module *module_from_buffer(py::object buffer, py::object py_options, 
      py::object message_handler)
  {
    const char *mod_buf;
    PYCUDA_BUFFER_SIZE_T len;
    if (PyObject_AsCharBuffer(buffer.ptr(), &mod_buf, &len))
      throw py::error_already_set();
    CUmodule mod;

#if CUDA_VERSION >= 2010
    const unsigned buf_size = 32768;
    char info_buf[buf_size], error_buf[buf_size];

    std::vector<CUjit_option> options;
    std::vector<void *> option_values;

#define ADD_OPTION_PTR(KEY, PTR) \
    { \
      options.push_back(KEY); \
      option_values.push_back(PTR); \
    }

    ADD_OPTION_PTR(CU_JIT_INFO_LOG_BUFFER, info_buf);
    ADD_OPTION_PTR(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, (void *) buf_size);
    ADD_OPTION_PTR(CU_JIT_ERROR_LOG_BUFFER, error_buf);
    ADD_OPTION_PTR(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, (void *) buf_size);

    PYTHON_FOREACH(key_value, py_options)
      ADD_OPTION_PTR(
          py::extract<CUjit_option>(key_value[0]), 
          (void *) py::extract<int>(key_value[1])());
#undef ADD_OPTION
    
    CUDAPP_PRINT_CALL_TRACE("cuModuleLoadDataEx");
    CUresult cu_status_code; \
    cu_status_code = cuModuleLoadDataEx(&mod, mod_buf, options.size(), 
         const_cast<CUjit_option *>(&*options.begin()),
         const_cast<void **>(&*option_values.begin()));

    size_t info_buf_size = size_t(option_values[1]);
    size_t error_buf_size = size_t(option_values[3]);

    if (message_handler != py::object())
      message_handler(cu_status_code == CUDA_SUCCESS,
          std::string(info_buf, info_buf_size),
          std::string(error_buf, error_buf_size));

    if (cu_status_code != CUDA_SUCCESS)
      throw cuda::error("cuModuleLoadDataEx", cu_status_code, 
          std::string(error_buf, error_buf_size).c_str());
#else
    if (py::len(py_options))
      throw cuda::error("module_from_buffer", CUDA_ERROR_INVALID_VALUE,
          "non-empty options argument only supported on CUDA 2.1 and newer");

    CUDAPP_CALL_GUARDED(cuModuleLoadData, (&mod, mod_buf));
#endif

    return new module(mod);
  }




  PyObject *device_allocation_to_long(device_allocation const &da)
  {
    return PyLong_FromUnsignedLong((CUdeviceptr) da);
  }




  py::handle<> pagelocked_empty(py::object shape, py::object dtype, 
      py::object order_py, unsigned mem_flags)
  {
    PyArray_Descr *tp_descr;
    if (PyArray_DescrConverter(dtype.ptr(), &tp_descr) != NPY_SUCCEED)
      throw py::error_already_set();

    py::extract<npy_intp> shape_as_int(shape);
    std::vector<npy_intp> dims;

    if (shape_as_int.check())
      dims.push_back(shape_as_int());
    else
      std::copy(
          py::stl_input_iterator<npy_intp>(shape),
          py::stl_input_iterator<npy_intp>(),
          back_inserter(dims));

    std::auto_ptr<host_allocation> alloc(
        new host_allocation(
          tp_descr->elsize*pycuda::size_from_dims(dims.size(), &dims.front()),
          mem_flags)
        );

    NPY_ORDER order = PyArray_CORDER;
    PyArray_OrderConverter(order_py.ptr(), &order);

    int ary_flags = 0;
    if (order == PyArray_FORTRANORDER)
      ary_flags |= NPY_FARRAY;
    else if (order == PyArray_CORDER)
      ary_flags |= NPY_CARRAY;
    else
      throw std::runtime_error("unrecognized order specifier");

    py::handle<> result = py::handle<>(PyArray_NewFromDescr(
        &PyArray_Type, tp_descr,
        dims.size(), &dims.front(), /*strides*/ NULL,
        alloc->data(), ary_flags, /*obj*/NULL));

    py::handle<> alloc_py(handle_from_new_ptr(alloc.release()));
    PyArray_BASE(result.get()) = alloc_py.get();
    Py_INCREF(alloc_py.get());

    return result;
  }




  void py_memcpy_htod(CUdeviceptr dst, py::object src)
  {
    const void *buf;
    PYCUDA_BUFFER_SIZE_T len;
    if (PyObject_AsReadBuffer(src.ptr(), &buf, &len))
      throw py::error_already_set();

    CUDAPP_CALL_GUARDED(cuMemcpyHtoD, (dst, buf, len));
  }




  void py_memcpy_htod_async(CUdeviceptr dst, py::object src, py::object stream_py)
  {
    const void *buf;
    PYCUDA_BUFFER_SIZE_T len;
    if (PyObject_AsReadBuffer(src.ptr(), &buf, &len))
      throw py::error_already_set();

    CUstream s_handle;
    if (stream_py.ptr() != Py_None)
    {
      const stream &s = py::extract<const stream &>(stream_py);
      s_handle = s.handle();
    }
    else
      s_handle = 0;

    CUDAPP_CALL_GUARDED(cuMemcpyHtoDAsync, (dst, buf, len, s_handle));
  }




  void py_memcpy_dtoh(py::object dest, CUdeviceptr src)
  {
    void *buf;
    PYCUDA_BUFFER_SIZE_T len;
    if (PyObject_AsWriteBuffer(dest.ptr(), &buf, &len))
      throw py::error_already_set();

    CUDAPP_CALL_GUARDED(cuMemcpyDtoH, (buf, src, len));
  }




  void py_memcpy_dtoh_async(py::object dest, CUdeviceptr src, py::object stream_py)
  {
    void *buf;
    PYCUDA_BUFFER_SIZE_T len;
    if (PyObject_AsWriteBuffer(dest.ptr(), &buf, &len))
      throw py::error_already_set();

    CUstream s_handle;
    if (stream_py.ptr() != Py_None)
    {
      const stream &s = py::extract<const stream &>(stream_py);
      s_handle = s.handle();
    }
    else
      s_handle = 0;

    CUDAPP_CALL_GUARDED(cuMemcpyDtoHAsync, (buf, src, len, s_handle));
  }




  void py_memcpy_htoa(array const &ary, unsigned int index, py::object src)
  {
    const void *buf;
    PYCUDA_BUFFER_SIZE_T len;
    if (PyObject_AsReadBuffer(src.ptr(), &buf, &len))
      throw py::error_already_set();

    CUDAPP_CALL_GUARDED(cuMemcpyHtoA, (ary.handle(), index, buf, len));
  }




  void py_memcpy_atoh(py::object dst, array const &ary, unsigned int index)
  {
    void *buf;
    PYCUDA_BUFFER_SIZE_T len;
    if (PyObject_AsWriteBuffer(dst.ptr(), &buf, &len))
      throw py::error_already_set();

    CUDAPP_CALL_GUARDED(cuMemcpyAtoH, (buf, ary.handle(), index, len));
  }

  bool have_gl_ext()
  {
#ifdef HAVE_GL
    return true;
#else
    return false;
#endif
  }
}




void pycuda_expose_tools();
void pycuda_expose_gl();




BOOST_PYTHON_MODULE(_driver)
{
  py::def("get_version", cuda_version);
#if CUDA_VERSION >= 2020
  py::def("get_driver_version", cuda::get_driver_version);
#endif

#define DECLARE_EXC(NAME, BASE) \
  Cuda##NAME = py::handle<>(PyErr_NewException("pycuda._driver." #NAME, BASE, NULL)); \
  py::scope().attr(#NAME) = Cuda##NAME;

  {
    DECLARE_EXC(Error, NULL);
    py::tuple memerr_bases = py::make_tuple(
        CudaError, 
        py::handle<>(py::borrowed(PyExc_MemoryError)));
    DECLARE_EXC(MemoryError, memerr_bases.ptr());
    DECLARE_EXC(LogicError, CudaError.get());
    DECLARE_EXC(LaunchError, CudaError.get());
    DECLARE_EXC(RuntimeError, CudaError.get());

    py::register_exception_translator<cuda::error>(translate_cuda_error);
  }

#if CUDA_VERSION >= 2000
  py::enum_<CUctx_flags>("ctx_flags")
    .value("SCHED_AUTO", CU_CTX_SCHED_AUTO)
    .value("SCHED_SPIN", CU_CTX_SCHED_SPIN)
    .value("SCHED_YIELD", CU_CTX_SCHED_YIELD)
    .value("SCHED_MASK", CU_CTX_SCHED_MASK)
#if CUDA_VERSION >= 2020
    .value("BLOCKING_SYNC", CU_CTX_BLOCKING_SYNC)
    .value("MAP_HOST", CU_CTX_MAP_HOST)
#endif
    .value("FLAGS_MASK", CU_CTX_FLAGS_MASK)
    ;
#endif

#if CUDA_VERSION >= 2020
  py::enum_<CUevent_flags>("event_flags")
    .value("DEFAULT", CU_EVENT_DEFAULT)
    .value("BLOCKING_SYNC", CU_EVENT_BLOCKING_SYNC)
    ;
#endif


  py::enum_<CUarray_format>("array_format")
    .value("UNSIGNED_INT8", CU_AD_FORMAT_UNSIGNED_INT8)
    .value("UNSIGNED_INT16", CU_AD_FORMAT_UNSIGNED_INT16)
    .value("UNSIGNED_INT32", CU_AD_FORMAT_UNSIGNED_INT32)
    .value("SIGNED_INT8"   , CU_AD_FORMAT_SIGNED_INT8)
    .value("SIGNED_INT16"  , CU_AD_FORMAT_SIGNED_INT16)
    .value("SIGNED_INT32"  , CU_AD_FORMAT_SIGNED_INT32)
    .value("HALF"          , CU_AD_FORMAT_HALF)
    .value("FLOAT"         , CU_AD_FORMAT_FLOAT)
    ;

#if CUDA_VERSION >= 3000
  {
    py::class_<array3d_flags> cls("array3d_flags", py::no_init);
    cls.attr("ARRAY3D_2DARRAY") = CUDA_ARRAY3D_2DARRAY;
  }
#endif

  py::enum_<CUaddress_mode>("address_mode")
    .value("WRAP", CU_TR_ADDRESS_MODE_WRAP)
    .value("CLAMP", CU_TR_ADDRESS_MODE_CLAMP)
    .value("MIRROR", CU_TR_ADDRESS_MODE_MIRROR)
    ;

  py::enum_<CUfilter_mode>("filter_mode")
    .value("POINT", CU_TR_FILTER_MODE_POINT)
    .value("LINEAR", CU_TR_FILTER_MODE_LINEAR)
    ;
  py::enum_<CUdevice_attribute>("device_attribute")
    .value("MAX_THREADS_PER_BLOCK", CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    .value("MAX_BLOCK_DIM_X", CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
    .value("MAX_BLOCK_DIM_Y", CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)
    .value("MAX_BLOCK_DIM_Z", CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)
    .value("MAX_GRID_DIM_X", CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)
    .value("MAX_GRID_DIM_Y", CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)
    .value("MAX_GRID_DIM_Z", CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)
#if CUDA_VERSION >= 2000
    .value("MAX_SHARED_MEMORY_PER_BLOCK", CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
#endif
    .value("SHARED_MEMORY_PER_BLOCK", CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK)
    .value("TOTAL_CONSTANT_MEMORY", CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY)
    .value("WARP_SIZE", CU_DEVICE_ATTRIBUTE_WARP_SIZE)
    .value("MAX_PITCH", CU_DEVICE_ATTRIBUTE_MAX_PITCH)
#if CUDA_VERSION >= 2000
    .value("MAX_REGISTERS_PER_BLOCK", CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK)
#endif
    .value("REGISTERS_PER_BLOCK", CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK)
    .value("CLOCK_RATE", CU_DEVICE_ATTRIBUTE_CLOCK_RATE)
    .value("TEXTURE_ALIGNMENT", CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT)

    .value("GPU_OVERLAP", CU_DEVICE_ATTRIBUTE_GPU_OVERLAP)
#if CUDA_VERSION >= 2000
    .value("MULTIPROCESSOR_COUNT", CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
#endif
#if CUDA_VERSION >= 2020
    .value("KERNEL_EXEC_TIMEOUT", CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT)
    .value("INTEGRATED", CU_DEVICE_ATTRIBUTE_INTEGRATED)
    .value("CAN_MAP_HOST_MEMORY", CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY)
    .value("COMPUTE_MODE", CU_DEVICE_ATTRIBUTE_COMPUTE_MODE)
#endif
#if CUDA_VERSION >= 3000
    .value("MAXIMUM_TEXTURE1D_WIDTH", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH)
    .value("MAXIMUM_TEXTURE2D_WIDTH", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH)
    .value("MAXIMUM_TEXTURE2D_HEIGHT", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT)
    .value("MAXIMUM_TEXTURE3D_WIDTH", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH)
    .value("MAXIMUM_TEXTURE3D_HEIGHT", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT)
    .value("MAXIMUM_TEXTURE3D_DEPTH", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH)
    .value("MAXIMUM_TEXTURE2D_ARRAY_WIDTH", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH)
    .value("MAXIMUM_TEXTURE2D_ARRAY_HEIGHT", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT)
    .value("MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES)
#ifdef CUDAPP_POST_30_BETA
    .value("SURFACE_ALIGNMENT", CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT)
    .value("CONCURRENT_KERNELS", CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS)
    .value("ECC_ENABLED", CU_DEVICE_ATTRIBUTE_ECC_ENABLED)
#endif
#endif
    ;

#if CUDA_VERSION >= 3000 && defined(CUDAPP_POST_30_BETA)
  py::enum_<CUfunc_cache_enum>("func_cache")
    .value("PREFER_NONE", CU_FUNC_CACHE_PREFER_NONE)
    .value("PREFER_SHARED", CU_FUNC_CACHE_PREFER_SHARED)
    .value("PREFER_L1", CU_FUNC_CACHE_PREFER_L1)
    ;
#endif

#if CUDA_VERSION >= 2020
  py::enum_<CUfunction_attribute>("function_attribute")
    .value("MAX_THREADS_PER_BLOCK", CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    .value("SHARED_SIZE_BYTES", CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)
    .value("CONST_SIZE_BYTES", CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES)
    .value("LOCAL_SIZE_BYTES", CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)
    .value("NUM_REGS", CU_FUNC_ATTRIBUTE_NUM_REGS)
#if CUDA_VERSION >= 3000 && defined(CUDAPP_POST_30_BETA)
    .value("PTX_VERSION", CU_FUNC_ATTRIBUTE_PTX_VERSION)
    .value("BINARY_VERSION", CU_FUNC_ATTRIBUTE_BINARY_VERSION)
#endif
    .value("MAX", CU_FUNC_ATTRIBUTE_MAX)
    ;
#endif

  py::enum_<CUmemorytype>("memory_type")
    .value("HOST", CU_MEMORYTYPE_HOST)
    .value("DEVICE", CU_MEMORYTYPE_DEVICE)
    .value("ARRAY", CU_MEMORYTYPE_ARRAY)
    ;

#if CUDA_VERSION >= 2020
  py::enum_<CUcomputemode>("compute_mode")
    .value("DEFAULT", CU_COMPUTEMODE_DEFAULT)
    .value("EXCLUSIVE", CU_COMPUTEMODE_EXCLUSIVE)
    .value("PROHIBITED", CU_COMPUTEMODE_PROHIBITED)
    ;
#endif

#if CUDA_VERSION >= 2010
  py::enum_<CUjit_option>("jit_option")
    .value("MAX_REGISTERS", CU_JIT_MAX_REGISTERS)
    .value("THREADS_PER_BLOCK", CU_JIT_THREADS_PER_BLOCK)
    .value("WALL_TIME", CU_JIT_WALL_TIME)
    .value("INFO_LOG_BUFFER", CU_JIT_INFO_LOG_BUFFER)
    .value("INFO_LOG_BUFFER_SIZE_BYTES", CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES)
    .value("ERROR_LOG_BUFFER", CU_JIT_ERROR_LOG_BUFFER)
    .value("ERROR_LOG_BUFFER_SIZE_BYTES", CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)
    .value("OPTIMIZATION_LEVEL", CU_JIT_OPTIMIZATION_LEVEL)
    .value("TARGET_FROM_CUCONTEXT", CU_JIT_TARGET_FROM_CUCONTEXT)
    .value("TARGET", CU_JIT_TARGET)
    .value("FALLBACK_STRATEGY", CU_JIT_FALLBACK_STRATEGY)
    ;

  py::enum_<CUjit_target>("jit_target")
    .value("COMPUTE_10", CU_TARGET_COMPUTE_10)
    .value("COMPUTE_11", CU_TARGET_COMPUTE_11)
    .value("COMPUTE_12", CU_TARGET_COMPUTE_12)
    .value("COMPUTE_13", CU_TARGET_COMPUTE_13)
#if CUDA_VERSION >= 3000
    .value("COMPUTE_20", CU_TARGET_COMPUTE_20)
#endif
    ;

  py::enum_<CUjit_fallback>("jit_fallback")
    .value("PREFER_PTX", CU_PREFER_PTX)
    .value("PREFER_BINARY", CU_PREFER_BINARY)
    ;
#endif

#if CUDA_VERSION >= 2020
  {
    py::class_<host_alloc_flags> cls("host_alloc_flags", py::no_init);
    cls.attr("PORTABLE") = CU_MEMHOSTALLOC_PORTABLE;
    cls.attr("DEVICEMAP") = CU_MEMHOSTALLOC_DEVICEMAP;
    cls.attr("WRITECOMBINED") = CU_MEMHOSTALLOC_WRITECOMBINED;
  }
#endif

  // graphics enums -----------------------------------------------------------
#if CUDA_VERSION >= 3000
  py::enum_<CUgraphicsRegisterFlags>("graphics_register_flags")
    .value("NONE", CU_GRAPHICS_REGISTER_FLAGS_NONE)
    ;

  py::enum_<CUgraphicsMapResourceFlags>("graphics_map_resource_flags")
    .value("NONE", CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)
    .value("READ_ONLY", CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY)
    .value("WRITE_DISCARD", CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD)
    ;

  py::enum_<CUarray_cubemap_face_enum>("array_cubemap_face")
    .value("POSITIVE_X", CU_CUBEMAP_FACE_POSITIVE_X)
    .value("NEGATIVE_X", CU_CUBEMAP_FACE_NEGATIVE_X)
    .value("POSITIVE_Y", CU_CUBEMAP_FACE_POSITIVE_Y)
    .value("NEGATIVE_Y", CU_CUBEMAP_FACE_NEGATIVE_Y)
    .value("POSITIVE_Z", CU_CUBEMAP_FACE_POSITIVE_Z)
    .value("NEGATIVE_Z", CU_CUBEMAP_FACE_NEGATIVE_Z)
    ;
#endif


  py::def("init", init,
      py::arg("flags")=0);

  {
    typedef device cl;
    py::class_<cl>("Device", py::no_init)
      .def("__init__", py::make_constructor(make_device))
      .DEF_SIMPLE_METHOD(count)
      .staticmethod("count")
      .DEF_SIMPLE_METHOD(name)
      .DEF_SIMPLE_METHOD(compute_capability)
      .DEF_SIMPLE_METHOD(total_memory)
      .def("get_attribute", device_get_attribute)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__hash__", &cl::hash)
      .def("make_context", &cl::make_context, 
          (py::args("self"), py::args("flags")=0))
      ;
  }

  {
    typedef context cl;
    py::class_<cl, shared_ptr<cl>, boost::noncopyable >("Context", py::no_init)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__hash__", &cl::hash)

      .DEF_SIMPLE_METHOD(detach)

#if CUDA_VERSION >= 2000
      .def("push", context_push)
      .DEF_SIMPLE_METHOD(pop)
      .staticmethod("pop")
      .DEF_SIMPLE_METHOD(get_device)
      .staticmethod("get_device")
#endif

      .DEF_SIMPLE_METHOD(synchronize)
      .staticmethod("synchronize")

      .def("get_current", (boost::shared_ptr<cl> (*)()) &cl::current_context)
      .staticmethod("get_current")
      ;
  }

  {
    typedef stream cl;
    py::class_<cl, boost::noncopyable>
      ("Stream", py::init<unsigned int>(py::arg("flags")=0))
      .DEF_SIMPLE_METHOD(synchronize)
      .DEF_SIMPLE_METHOD(is_done)
      ;
  }

  {
    typedef module cl;
    py::class_<cl, boost::noncopyable, shared_ptr<cl> >("Module", py::no_init)
      .def("get_function", &cl::get_function, (py::args("self", "name")),
          py::with_custodian_and_ward_postcall<0, 1>())
      .def("get_global", &cl::get_global, (py::args("self", "name")))
      .def("get_texref", module_get_texref, 
          (py::args("self", "name")),
          py::return_value_policy<py::manage_new_object>())
      ;
  }

  py::def("module_from_file", module_from_file, (py::arg("filename")),
      py::return_value_policy<py::manage_new_object>());
  py::def("module_from_buffer", module_from_buffer, 
      (py::arg("buffer"), 
       py::arg("options")=py::list(), 
       py::arg("message_handler")=py::object()),
      py::return_value_policy<py::manage_new_object>());

  {
    typedef function cl;
    py::class_<cl>("Function", py::no_init)
      .DEF_SIMPLE_METHOD(set_block_shape)
      .DEF_SIMPLE_METHOD(set_shared_size)
      .DEF_SIMPLE_METHOD(param_set_size)
      .def("param_seti", (void (cl::*)(int, unsigned int)) &cl::param_set)
      .def("param_setf", (void (cl::*)(int, float )) &cl::param_set)
      .def("param_setv", function_param_setv)
      .DEF_SIMPLE_METHOD(param_set_texref)

      .DEF_SIMPLE_METHOD(launch)
      .DEF_SIMPLE_METHOD(launch_grid)
      .DEF_SIMPLE_METHOD(launch_grid_async)

#if CUDA_VERSION >= 2020
      .DEF_SIMPLE_METHOD(get_attribute)
#endif
#if CUDA_VERSION >= 3000 && defined(CUDAPP_POST_30_BETA)
      .DEF_SIMPLE_METHOD(set_cache_config)
#endif
      ;
  }

  {
    typedef device_allocation cl;
    py::class_<cl, boost::noncopyable>("DeviceAllocation", py::no_init)
      .def("__int__", &cl::operator CUdeviceptr)
      .def("__long__", device_allocation_to_long)
      .DEF_SIMPLE_METHOD(free)
      ;

    py::implicitly_convertible<device_allocation, CUdeviceptr>();
  }

  {
    typedef host_allocation cl;
    py::class_<cl, boost::noncopyable>("HostAllocation", py::no_init)
      .DEF_SIMPLE_METHOD(free)
#if CUDA_VERSION >= 2020
      .DEF_SIMPLE_METHOD(get_device_pointer)
#endif
      ;
  }

  DEF_SIMPLE_FUNCTION(mem_get_info);
  py::def("mem_alloc", mem_alloc_wrap, 
      py::return_value_policy<py::manage_new_object>());
  py::def("mem_alloc_pitch", mem_alloc_pitch_wrap,
      py::args("width", "height", "access_size"));
  DEF_SIMPLE_FUNCTION(mem_get_address_range);

  py::def("memset_d8",  py_memset_d8, py::args("dest", "data", "size"));
  py::def("memset_d16", py_memset_d16, py::args("dest", "data", "size"));
  py::def("memset_d32", py_memset_d32, py::args("dest", "data", "size"));

  py::def("memset_d2d8", py_memset_d2d8, 
      py::args("dest", "pitch", "data", "width", "height"));
  py::def("memset_d2d16", py_memset_d2d16, 
      py::args("dest", "pitch", "data", "width", "height"));
  py::def("memset_d2d32", py_memset_d2d32, 
      py::args("dest", "pitch", "data", "width", "height"));

  py::def("memcpy_htod", py_memcpy_htod, 
      (py::args("dest"), py::arg("src")));
  py::def("memcpy_htod_async", py_memcpy_htod_async, 
      (py::args("dest"), py::arg("src"), py::arg("stream")=py::object()));
  py::def("memcpy_dtoh", py_memcpy_dtoh, 
      (py::args("dest"), py::arg("src")));
  py::def("memcpy_dtoh_async", py_memcpy_dtoh_async, 
      (py::args("dest"), py::arg("src"), py::arg("stream")=py::object()));

  py::def("memcpy_dtod", py_memcpy_dtod, py::args("dest", "src", "size"));
#if CUDA_VERSION >= 3000
  py::def("memcpy_dtod_async", py_memcpy_dtod_async, 
      (py::args("dest", "src", "size"), py::arg("stream")=py::object()));
#endif

  DEF_SIMPLE_FUNCTION_WITH_ARGS(memcpy_dtoa,
      ("ary", "index", "src", "len"));
  DEF_SIMPLE_FUNCTION_WITH_ARGS(memcpy_atod,
      ("dest", "ary", "index", "len"));
  DEF_SIMPLE_FUNCTION_WITH_ARGS(py_memcpy_htoa,
      ("ary", "index", "src"));
  DEF_SIMPLE_FUNCTION_WITH_ARGS(py_memcpy_atoh,
      ("dest", "ary", "index"));

  DEF_SIMPLE_FUNCTION_WITH_ARGS(memcpy_atoa,
      ("dest", "dest_index", "src", "src_index", "len"));

  {
    typedef memcpy_2d cl;
    py::class_<cl>("Memcpy2D")
      .def_readwrite("src_x_in_bytes", &cl::srcXInBytes)
      .def_readwrite("src_y", &cl::srcY)
      .def_readwrite("src_memory_type", &cl::srcMemoryType)
      .def_readwrite("src_device", &cl::srcDevice)
      .def_readwrite("src_pitch", &cl::srcPitch)

      .DEF_SIMPLE_METHOD(set_src_host)
      .DEF_SIMPLE_METHOD(set_src_array)
      .DEF_SIMPLE_METHOD(set_src_device)

      .def_readwrite("dst_x_in_bytes", &cl::dstXInBytes)
      .def_readwrite("dst_y", &cl::dstY)
      .def_readwrite("dst_memory_type", &cl::dstMemoryType)
      .def_readwrite("dst_device", &cl::dstDevice)
      .def_readwrite("dst_pitch", &cl::dstPitch)

      .DEF_SIMPLE_METHOD(set_dst_host)
      .DEF_SIMPLE_METHOD(set_dst_array)
      .DEF_SIMPLE_METHOD(set_dst_device)

      .def_readwrite("width_in_bytes", &cl::WidthInBytes)
      .def_readwrite("height", &cl::Height)

      .def("__call__", &cl::execute, py::args("self", "aligned"))
      .def("__call__", &cl::execute_async)
      ;
  }

#if CUDA_VERSION >= 2000
  {
    typedef memcpy_3d cl;
    py::class_<cl>("Memcpy3D")
      .def_readwrite("src_x_in_bytes", &cl::srcXInBytes)
      .def_readwrite("src_y", &cl::srcY)
      .def_readwrite("src_z", &cl::srcZ)
      .def_readwrite("src_lod", &cl::srcLOD)
      .def_readwrite("src_memory_type", &cl::srcMemoryType)
      .def_readwrite("src_device", &cl::srcDevice)
      .def_readwrite("src_pitch", &cl::srcPitch)
      .def_readwrite("src_height", &cl::srcHeight)

      .DEF_SIMPLE_METHOD(set_src_host)
      .DEF_SIMPLE_METHOD(set_src_array)
      .DEF_SIMPLE_METHOD(set_src_device)

      .def_readwrite("dst_x_in_bytes", &cl::dstXInBytes)
      .def_readwrite("dst_y", &cl::dstY)
      .def_readwrite("dst_z", &cl::dstZ)
      .def_readwrite("dst_lod", &cl::dstLOD)
      .def_readwrite("dst_memory_type", &cl::dstMemoryType)
      .def_readwrite("dst_device", &cl::dstDevice)
      .def_readwrite("dst_pitch", &cl::dstPitch)
      .def_readwrite("dst_height", &cl::dstHeight)

      .DEF_SIMPLE_METHOD(set_dst_host)
      .DEF_SIMPLE_METHOD(set_dst_array)
      .DEF_SIMPLE_METHOD(set_dst_device)

      .def_readwrite("width_in_bytes", &cl::WidthInBytes)
      .def_readwrite("height", &cl::Height)
      .def_readwrite("depth", &cl::Depth)

      .def("__call__", &cl::execute)
      .def("__call__", &cl::execute_async)
      ;
  }
#endif

  py::def("pagelocked_empty", pagelocked_empty,
      (py::arg("shape"), py::arg("dtype"), py::arg("order")="C",
       py::arg("mem_flags")=0));

  {
    typedef event cl;
    py::class_<cl, boost::noncopyable>
      ("Event", py::init<py::optional<unsigned int> >(py::arg("flags")))
      .def("record", &cl::record,
          py::arg("stream")=py::object(), py::return_self<>())
      .def("synchronize", &cl::synchronize, py::return_self<>())
      .DEF_SIMPLE_METHOD(query)
      .DEF_SIMPLE_METHOD(time_since)
      .DEF_SIMPLE_METHOD(time_till)
      ;
  }

  {
    typedef CUDA_ARRAY_DESCRIPTOR cl;
    py::class_<cl>("ArrayDescriptor")
      .def_readwrite("width", &cl::Width)
      .def_readwrite("height", &cl::Height)
      .def_readwrite("format", &cl::Format)
      .def_readwrite("num_channels", &cl::NumChannels)
      ;
  }

#if CUDA_VERSION >= 2000
  {
    typedef CUDA_ARRAY3D_DESCRIPTOR cl;
    py::class_<cl>("ArrayDescriptor3D")
      .def_readwrite("width", &cl::Width)
      .def_readwrite("height", &cl::Height)
      .def_readwrite("depth", &cl::Depth)
      .def_readwrite("format", &cl::Format)
      .def_readwrite("num_channels", &cl::NumChannels)
      ;
  }
#endif

  {
    typedef array cl;
    py::class_<cl, shared_ptr<cl>, boost::noncopyable>
      ("Array", py::init<const CUDA_ARRAY_DESCRIPTOR &>())
      .DEF_SIMPLE_METHOD(free)
      .DEF_SIMPLE_METHOD(get_descriptor)
#if CUDA_VERSION >= 2000
      .def(py::init<const CUDA_ARRAY3D_DESCRIPTOR &>())
      .DEF_SIMPLE_METHOD(get_descriptor_3d)
#endif
      ;
  }

  {
    typedef texture_reference cl;
    py::class_<cl, boost::noncopyable>("TextureReference")
      .DEF_SIMPLE_METHOD(set_array)
      .def("set_address", &cl::set_address, 
          (py::arg("devptr"), py::arg("bytes"), py::arg("allow_offset")=false))
#if CUDA_VERSION >= 2020
      .DEF_SIMPLE_METHOD_WITH_ARGS(set_address_2d, ("devptr", "descr", "pitch"))
#endif
      .DEF_SIMPLE_METHOD_WITH_ARGS(set_format, ("format", "num_components"))
      .DEF_SIMPLE_METHOD_WITH_ARGS(set_address_mode, ("dim", "am"))
      .DEF_SIMPLE_METHOD(set_filter_mode)
      .DEF_SIMPLE_METHOD(set_flags)
      .DEF_SIMPLE_METHOD(get_address)
      .def("get_array", &cl::get_array,
          py::return_value_policy<py::manage_new_object>())
      .DEF_SIMPLE_METHOD(get_address_mode)
      .DEF_SIMPLE_METHOD(get_filter_mode)

#if CUDA_VERSION >= 2000
      .DEF_SIMPLE_METHOD(get_format)
#endif

      .DEF_SIMPLE_METHOD(get_flags)
      ;
  }

  py::scope().attr("TRSA_OVERRIDE_FORMAT") = CU_TRSA_OVERRIDE_FORMAT;
  py::scope().attr("TRSF_READ_AS_INTEGER") = CU_TRSF_READ_AS_INTEGER;
  py::scope().attr("TRSF_NORMALIZED_COORDINATES") = CU_TRSF_NORMALIZED_COORDINATES;
  py::scope().attr("TR_DEFAULT") = CU_PARAM_TR_DEFAULT;

  DEF_SIMPLE_FUNCTION(have_gl_ext);

  pycuda_expose_tools();
#ifdef HAVE_GL
  pycuda_expose_gl();
#endif
}
