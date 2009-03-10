#include "cuda.hpp"

boost::thread_specific_ptr<cuda::context_stack_t> cuda::context_stack_ptr;
