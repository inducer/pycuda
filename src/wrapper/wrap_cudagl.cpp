#include <cuda.hpp>
#include <cuda_gl.hpp>

#include "tools.hpp"
#include "wrap_helpers.hpp"




using namespace cuda;
using namespace cuda::gl;
using boost::shared_ptr;




void pycuda_expose_gl()
{
  using py::arg;
  using py::args;

  DEF_SIMPLE_FUNCTION(gl_init);

  py::def("make_gl_context", make_gl_context, (arg("dev"), arg("flags")=0));

  py::enum_<CUgraphicsMapResourceFlags>("map_flags")
    .value("CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE", CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)
    .value("CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY", CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY)
    .value("CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD", CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD)
  ;

  py::enum_<GLenum>("target_flags")
    .value("GL_TEXTURE_2D", GL_TEXTURE_2D)
    .value("GL_TEXTURE_RECTANGLE", GL_TEXTURE_RECTANGLE)
    .value("GL_TEXTURE_CUBE_MAP", GL_TEXTURE_CUBE_MAP)
    .value("GL_TEXTURE_3D", GL_TEXTURE_3D)
    .value("GL_TEXTURE_2D_ARRAY", GL_TEXTURE_2D_ARRAY)
    .value("GL_RENDERBUFFER", GL_RENDERBUFFER)
  ;

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

  {
    typedef registered_buffer cl;
    py::class_<cl, shared_ptr<cl> >("RegisteredBuffer", py::init<GLuint, py::optional<CUgraphicsMapResourceFlags> >())
      .DEF_SIMPLE_METHOD(handle)
      .DEF_SIMPLE_METHOD(unregister)
      .def("map", map_registered_object,
          py::return_value_policy<py::manage_new_object>())
      ;
  }

  {
    typedef registered_image cl;
    py::class_<cl, shared_ptr<cl> >("RegisteredImage", py::init<GLuint, GLenum, py::optional<CUgraphicsMapResourceFlags> >())
      .DEF_SIMPLE_METHOD(handle)
      .DEF_SIMPLE_METHOD(unregister)
      .def("map", map_registered_object,
          py::return_value_policy<py::manage_new_object>())
      ;
  }

  {
    typedef registered_mapping cl;
    py::class_<cl>("RegisteredMapping", py::no_init)
      .DEF_SIMPLE_METHOD(unmap)
      .DEF_SIMPLE_METHOD(device_ptr)
      .DEF_SIMPLE_METHOD(size)
      ;
  }
}
