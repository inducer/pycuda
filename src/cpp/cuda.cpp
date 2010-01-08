#include "cuda.hpp"

boost::thread_specific_ptr<cuda::context_stack> cuda::context_stack_ptr;
