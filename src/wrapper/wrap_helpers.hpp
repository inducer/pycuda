#ifndef PYCUDA_WRAP_HELPERS_HEADER_SEEN
#define PYCUDA_WRAP_HELPERS_HEADER_SEEN




#define PYTHON_ERROR(TYPE, REASON) \
{ \
  PyErr_SetString(PyExc_##TYPE, REASON); \
  throw boost::python::error_already_set(); \
}





#endif
