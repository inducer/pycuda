#include <cuda.hpp>

#include <utility>
#include <numeric>
#include <algorithm>

#include "tools.hpp"
#include "wrap_helpers.hpp"
#include <boost/python/stl_iterator.hpp>




#if CUDA_VERSION < 1010
#error PyCuda only works with CUDA 1.1 or newer
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




  void  _cuMemsetD8( CUdeviceptr dstDevice, unsigned char uc, unsigned int N ) 
  { CUDAPP_CALL_GUARDED_THREADED(cuMemsetD8, (dstDevice, uc, N )); }
  void  _cuMemsetD16( CUdeviceptr dstDevice, unsigned short us, unsigned int N ) 
  { CUDAPP_CALL_GUARDED_THREADED(cuMemsetD16, (dstDevice, us, N )); }
  void  _cuMemsetD32( CUdeviceptr dstDevice, unsigned int ui, unsigned int N ) 
  { CUDAPP_CALL_GUARDED_THREADED(cuMemsetD32, (dstDevice, ui, N )); }

  void  _cuMemsetD2D8( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height )
  { CUDAPP_CALL_GUARDED_THREADED(cuMemsetD2D8, (dstDevice, dstPitch, uc, Width, Height)); }

  void  _cuMemsetD2D16( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height )
  { CUDAPP_CALL_GUARDED_THREADED(cuMemsetD2D16, (dstDevice, dstPitch, us, Width, Height)); }

  void  _cuMemsetD2D32( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height )
  { CUDAPP_CALL_GUARDED_THREADED(cuMemsetD2D32, (dstDevice, dstPitch, ui, Width, Height)); }

  void  _cuMemcpyDtoD (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount )
  { CUDAPP_CALL_GUARDED_THREADED(cuMemcpyDtoD, (dstDevice, srcDevice, ByteCount)); }




  void function_param_setv(function &f, int offset, py::object buffer)
  { 
    const void *buf;
    PYCUDA_BUFFER_SIZE_T len;
    if (PyObject_AsReadBuffer(buffer.ptr(), &buf, &len))
      throw py::error_already_set();
    f.param_setv(offset, const_cast<void *>(buf), len);
  }




  module *module_from_buffer(py::object buffer)
  {
    const char *buf;
    PYCUDA_BUFFER_SIZE_T len;
    if (PyObject_AsCharBuffer(buffer.ptr(), &buf, &len))
      throw py::error_already_set();
    CUmodule mod;
    CUDAPP_CALL_GUARDED(cuModuleLoadData, (&mod, buf));
    return new module(mod);
  }




  PyObject *device_allocation_to_long(device_allocation const &da)
  {
    return PyLong_FromUnsignedLong((CUdeviceptr) da);
  }




  py::handle<> pagelocked_empty(py::object shape, py::object dtype, py::object order_py)
  {
    PyArray_Descr *tp_descr;
    if (PyArray_DescrConverter(dtype.ptr(), &tp_descr) != NPY_SUCCEED)
      throw py::error_already_set();

    std::vector<npy_intp> dims;
    std::copy(
        py::stl_input_iterator<npy_intp>(shape),
        py::stl_input_iterator<npy_intp>(),
        back_inserter(dims));

    std::auto_ptr<host_allocation> alloc(
        new host_allocation(
          tp_descr->elsize*pycuda::size_from_dims(dims.size(), &dims.front())));

    NPY_ORDER order = PyArray_CORDER;
    PyArray_OrderConverter(order_py.ptr(), &order);

    int flags = 0;
    if (order == PyArray_FORTRANORDER)
      flags |= NPY_FARRAY;
    else if (order == PyArray_CORDER)
      flags |= NPY_CARRAY;
    else
      throw std::runtime_error("unrecognized order specifier");

    py::handle<> result = py::handle<>(PyArray_NewFromDescr(
        &PyArray_Type, tp_descr,
        dims.size(), &dims.front(), /*strides*/ NULL,
        alloc->data(), flags, /*obj*/NULL));

    py::handle<> alloc_py(handle_from_new_ptr(alloc.release()));
    PyArray_BASE(result.get()) = alloc_py.get();
    Py_INCREF(alloc_py.get());

    return result;
  }




  void py_memcpy_htod(CUdeviceptr dst, py::object src, py::object stream_py)
  {
    const void *buf;
    PYCUDA_BUFFER_SIZE_T len;
    if (PyObject_AsReadBuffer(src.ptr(), &buf, &len))
      throw py::error_already_set();

    if (stream_py.ptr() == Py_None)
    {
      CUDAPP_CALL_GUARDED(cuMemcpyHtoD, (dst, buf, len));
    }
    else
    {
      const stream &s = py::extract<const stream &>(stream_py);
      CUDAPP_CALL_GUARDED(cuMemcpyHtoDAsync, (dst, buf, len, s.data()));
    }
  }




  void py_memcpy_dtoh(py::object dest, CUdeviceptr src, py::object stream_py)
  {
    void *buf;
    PYCUDA_BUFFER_SIZE_T len;
    if (PyObject_AsWriteBuffer(dest.ptr(), &buf, &len))
      throw py::error_already_set();

    if (stream_py.ptr() == Py_None)
    {
      CUDAPP_CALL_GUARDED(cuMemcpyDtoH, (buf, src, len));
    }
    else
    {
      const stream &s = py::extract<const stream &>(stream_py);
      CUDAPP_CALL_GUARDED(cuMemcpyDtoHAsync, (buf, src, len, s.data()));
    }
  }




  void py_memcpy_htoa(array const &ary, unsigned int index, py::object src)
  {
    const void *buf;
    PYCUDA_BUFFER_SIZE_T len;
    if (PyObject_AsReadBuffer(src.ptr(), &buf, &len))
      throw py::error_already_set();

    CUDAPP_CALL_GUARDED(cuMemcpyHtoA, (ary.data(), index, buf, len));
  }




  void py_memcpy_atoh(py::object dst, array const &ary, unsigned int index)
  {
    void *buf;
    PYCUDA_BUFFER_SIZE_T len;
    if (PyObject_AsWriteBuffer(dst.ptr(), &buf, &len))
      throw py::error_already_set();

    CUDAPP_CALL_GUARDED(cuMemcpyAtoH, (buf, ary.data(), index, len));
  }

}




void pycuda_expose_tools();




BOOST_PYTHON_MODULE(_driver)
{
#define DECLARE_EXC(NAME, BASE) \
  Cuda##NAME = py::handle<>(PyErr_NewException("pycuda._driver." #NAME, BASE, NULL)); \
  py::scope().attr(#NAME) = Cuda##NAME;

  def("get_version", cuda_version);

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
    .value("SCHED_FLAGS_MASK", CU_CTX_FLAGS_MASK)
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
    ;
  py::enum_<CUmemorytype>("memory_type")
    .value("HOST", CU_MEMORYTYPE_HOST)
    .value("DEVICE", CU_MEMORYTYPE_DEVICE)
    .value("ARRAY", CU_MEMORYTYPE_ARRAY)
    ;

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
      .DEF_SIMPLE_METHOD(get_attribute)
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
      .DEF_SIMPLE_METHOD(detach)

#if CUDA_VERSION >= 2000
      .def("push", context_push)
      .DEF_SIMPLE_METHOD(pop)
      .DEF_SIMPLE_METHOD(get_device)
      .staticmethod("get_device")
#endif

      .DEF_SIMPLE_METHOD(synchronize)
      .staticmethod("synchronize")
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
  py::def("module_from_buffer", module_from_buffer, (py::arg("buffer")),
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
      ;
  }

  DEF_SIMPLE_FUNCTION(mem_get_info);
  py::def("mem_alloc", mem_alloc_wrap, 
      py::return_value_policy<py::manage_new_object>());
  py::def("mem_alloc_pitch", mem_alloc_pitch_wrap,
      py::args("width", "height", "access_size"));
  DEF_SIMPLE_FUNCTION(mem_get_address_range);

  py::def("memset_d8", _cuMemsetD8, py::args("dest", "data", "size"));
  py::def("memset_d16", _cuMemsetD16, py::args("dest", "data", "size"));
  py::def("memset_d32", _cuMemsetD32, py::args("dest", "data", "size"));

  py::def("memset_d2d8", _cuMemsetD2D8, 
      py::args("dest", "pitch", "data", "width", "height"));
  py::def("memset_d2d16", _cuMemsetD2D16, 
      py::args("dest", "pitch", "data", "width", "height"));
  py::def("memset_d2d32", _cuMemsetD2D32, 
      py::args("dest", "pitch", "data", "width", "height"));

  py::def("memcpy_htod", py_memcpy_htod, 
      (py::args("dest"), py::arg("src"), py::arg("stream")=py::object()));
  py::def("memcpy_dtoh", py_memcpy_dtoh, 
      (py::args("dest"), py::arg("src"), py::arg("stream")=py::object()));
  py::def("memcpy_dtod", _cuMemcpyDtoD, py::args("dest", "src", "size"));

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
      (py::arg("shape"), py::arg("dtype"), py::arg("order")="C"));

  {
    typedef event cl;
    py::class_<cl, boost::noncopyable>
      ("Event", py::init<py::optional<unsigned int> >(py::arg("flags")))
      .DEF_SIMPLE_METHOD(record)
      .def("record", &cl::record_in_stream)
      .DEF_SIMPLE_METHOD(synchronize)
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
      .DEF_SIMPLE_METHOD_WITH_ARGS(set_address, ("devptr", "bytes"))
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

  pycuda_expose_tools();
}
