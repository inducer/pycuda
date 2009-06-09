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
}
