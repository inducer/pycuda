#ifndef PYCUDA_WRAP_HELPERS_HEADER_SEEN
#define PYCUDA_WRAP_HELPERS_HEADER_SEEN




#include <pybind11/pybind11.h>




#define ENUM_VALUE(NAME) \
  value(#NAME, NAME)

#define DEF_SIMPLE_METHOD(NAME) \
  def(#NAME, &cl::NAME)

#define DEF_SIMPLE_METHOD_WITH_ARGS(NAME, ARGS) \
  def(#NAME, &cl::NAME, boost::python::args ARGS)

#define DEF_SIMPLE_FUNCTION(NAME) \
  m.def(#NAME, &NAME)

#define DEF_SIMPLE_FUNCTION_WITH_ARGS(NAME, ARGS) \
  m.def(#NAME, &NAME, boost::python::args ARGS)

#define DEF_SIMPLE_RO_MEMBER(NAME) \
  def_readonly(#NAME, &cl::m_##NAME)

#define DEF_SIMPLE_RW_MEMBER(NAME) \
  def_readwrite(#NAME, &cl::m_##NAME)




namespace
{
  template <typename T>
  inline py::object handle_from_new_ptr(T *ptr)
  {
    return py::cast(ptr, py::return_value_policy::take_ownership);
  }
}




#endif
