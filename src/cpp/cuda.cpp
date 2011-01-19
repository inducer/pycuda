#include "cuda.hpp"

boost::thread_specific_ptr<pycuda::context_stack> pycuda::context_stack_ptr;
