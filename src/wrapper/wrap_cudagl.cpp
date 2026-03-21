#ifdef _WIN32
#include <windows.h>
#endif
#include <cuda.hpp>
#include <cuda_gl.hpp>

#include "tools.hpp"
#include "wrap_helpers.hpp"




using namespace pycuda;
using namespace pycuda::gl;
namespace py = pybind11;
using std::shared_ptr;




void pycuda_expose_gl(py::module_ &m)
{
  using py::arg;


  m.def("make_gl_context", make_gl_context, arg("dev"), arg("flags")=0);

  // {{{ new-style

#if CUDAPP_CUDA_VERSION >= 3000
  py::enum_<CUgraphicsMapResourceFlags>(m, "graphics_map_flags")
    .value("NONE", CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)
    .value("READ_ONLY", CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY)
    .value("WRITE_DISCARD", CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD)
  ;

  {
    typedef registered_object cl;
    py::class_<cl, shared_ptr<cl> >(m, "RegisteredObject")
      .DEF_SIMPLE_METHOD(gl_handle)
      .DEF_SIMPLE_METHOD(unregister)
      .def("map", map_registered_object,
          arg("stream")=py::none(),
          py::return_value_policy::take_ownership)
      ;
  }

  {
    typedef registered_buffer cl;
    py::class_<cl, registered_object, shared_ptr<cl> >(m, "RegisteredBuffer")
      .def(py::init<GLuint>())
      .def(py::init<GLuint, CUgraphicsMapResourceFlags>())
      ;
  }

  {
    typedef registered_image cl;
    py::class_<cl, registered_object, shared_ptr<cl> >(m, "RegisteredImage")
      .def(py::init<GLuint, GLenum>())
      .def(py::init<GLuint, GLenum, CUgraphicsMapResourceFlags>())
      ;
  }

  {
    typedef registered_mapping cl;
    py::class_<cl>(m, "RegisteredMapping")
      .def("unmap", &cl::unmap_no_strm)
      .def("unmap", &cl::unmap)
      .DEF_SIMPLE_METHOD(device_ptr_and_size)
      .def("array", &cl::array,
          arg("index"), arg("level"),
          py::return_value_policy::take_ownership)
      ;
  }
#endif

  // }}}

  // {{{ old-style

  DEF_SIMPLE_FUNCTION(gl_init);

  {
    typedef buffer_object cl;
    py::class_<cl, shared_ptr<cl> >(m, "BufferObject", py::init<GLuint>())
      .DEF_SIMPLE_METHOD(handle)
      .DEF_SIMPLE_METHOD(unregister)
      .def("map", map_buffer_object,
          py::return_value_policy::take_ownership)
      ;
  }

  {
    typedef buffer_object_mapping cl;
    py::class_<cl>(m, "BufferObjectMapping")
      .DEF_SIMPLE_METHOD(unmap)
      .DEF_SIMPLE_METHOD(device_ptr)
      .DEF_SIMPLE_METHOD(size)
      ;
  }

  // }}}

}

// vim: foldmethod=marker
