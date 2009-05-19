#ifndef PYCUDA_WRAP_HELPERS_HEADER_SEEN
#define PYCUDA_WRAP_HELPERS_HEADER_SEEN




#include <boost/python.hpp>
#include <boost/foreach.hpp>
#include <boost/python/stl_iterator.hpp>




#define PYTHON_ERROR(TYPE, REASON) \
{ \
  PyErr_SetString(PyExc_##TYPE, REASON); \
  throw boost::python::error_already_set(); \
}

#define ENUM_VALUE(NAME) \
  value(#NAME, NAME)

#define DEF_SIMPLE_METHOD(NAME) \
  def(#NAME, &cl::NAME)

#define DEF_SIMPLE_METHOD_WITH_ARGS(NAME, ARGS) \
  def(#NAME, &cl::NAME, boost::python::args ARGS)

#define DEF_SIMPLE_FUNCTION(NAME) \
  boost::python::def(#NAME, &NAME)

#define DEF_SIMPLE_FUNCTION_WITH_ARGS(NAME, ARGS) \
  boost::python::def(#NAME, &NAME, boost::python::args ARGS)

#define DEF_SIMPLE_RO_MEMBER(NAME) \
  def_readonly(#NAME, &cl::m_##NAME)

#define DEF_SIMPLE_RW_MEMBER(NAME) \
  def_readwrite(#NAME, &cl::m_##NAME)

#define PYTHON_FOREACH(NAME, ITERABLE) \
  BOOST_FOREACH(boost::python::object NAME, \
      std::make_pair( \
        boost::python::stl_input_iterator<boost::python::object>(ITERABLE), \
        boost::python::stl_input_iterator<boost::python::object>()))




namespace
{
  template <typename T>
  inline boost::python::handle<> handle_from_new_ptr(T *ptr)
  {
    return boost::python::handle<>(
        typename boost::python::manage_new_object::apply<T *>::type()(ptr));
  }
}




#endif
