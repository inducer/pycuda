#ifdef _WIN32
#include <windows.h>
#endif
#include <cuda.hpp>
#include <cuda_gl.hpp>

#include "tools.hpp"
#include "wrap_helpers.hpp"




using namespace pycuda;
using namespace pycuda::gl;
using boost::shared_ptr;




void pycuda_expose_gl()
{
  using py::arg;
  using py::args;


  py::def("make_gl_context", make_gl_context, (arg("dev"), arg("flags")=0));

  // {{{ new-style

#if CUDAPP_CUDA_VERSION >= 3000
  py::enum_<CUgraphicsMapResourceFlags>("graphics_map_flags")
    .value("NONE", CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)
    .value("READ_ONLY", CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY)
    .value("WRITE_DISCARD", CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD)
  ;

  {
    typedef registered_object cl;
    py::class_<cl, shared_ptr<cl> >("RegisteredObject", py::no_init)
      .DEF_SIMPLE_METHOD(gl_handle)
      .DEF_SIMPLE_METHOD(unregister)
      .def("map", map_registered_object,
          (arg("robj"), arg("stream")=py::object()),
          py::return_value_policy<py::manage_new_object>())
      ;
  }

  {
    typedef registered_buffer cl;
    py::class_<cl, shared_ptr<cl>, py::bases<registered_object> >(
        "RegisteredBuffer",
        py::init<GLuint, py::optional<CUgraphicsMapResourceFlags> >())
      ;
  }

  {
    typedef registered_image cl;
    py::class_<cl, shared_ptr<cl>, py::bases<registered_object> >(
        "RegisteredImage", 
        py::init<GLuint, GLenum, py::optional<CUgraphicsMapResourceFlags> >())
      ;
  }

  {
    typedef registered_mapping cl;
    py::class_<cl>("RegisteredMapping", py::no_init)
      .def("unmap", &cl::unmap_no_strm)
      .def("unmap", &cl::unmap)
      .DEF_SIMPLE_METHOD(device_ptr_and_size)
      .def("array", &cl::array,
          (py::args("self", "index", "level")),
          py::return_value_policy<py::manage_new_object>())
      ;
  }
#endif

  // }}}

  // {{{ old-style

  DEF_SIMPLE_FUNCTION(gl_init);

  {
    typedef buffer_object cl;
    py::class_<cl, shared_ptr<cl> >("BufferObject", py::init<GLuint>())
      .DEF_SIMPLE_METHOD(handle)
      .DEF_SIMPLE_METHOD(unregister)
      .def("map", map_buffer_object,
          py::return_value_policy<py::manage_new_object>())
      ;
  }

  {
    typedef buffer_object_mapping cl;
    py::class_<cl>("BufferObjectMapping", py::no_init)
      .DEF_SIMPLE_METHOD(unmap)
      .DEF_SIMPLE_METHOD(device_ptr)
      .DEF_SIMPLE_METHOD(size)
      ;
  }

  // }}}

}

// vim: foldmethod=marker
