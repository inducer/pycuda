#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pycuda_ARRAY_API

#include "cuda.hpp"

std::thread_specific_ptr<pycuda::context_stack> pycuda::context_stack_ptr;
