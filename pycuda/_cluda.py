CLUDA_PREAMBLE = """
#define local_barrier() __syncthreads();

#define WITHIN_KERNEL __device__
#define KERNEL extern "C" __global__
#define GLOBAL_MEM /* empty */
#define LOCAL_MEM __shared__
#define LOCAL_MEM_ARG /* empty */
#define REQD_WG_SIZE(X,Y,Z) __launch_bounds__(X*Y*Z, 1)

#define LID_0 threadIdx.x
#define LID_1 threadIdx.y
#define LID_2 threadIdx.z

#define GID_0 blockIdx.x
#define GID_1 blockIdx.y
#define GID_2 blockIdx.z

#define LDIM_0 blockDim.x
#define LDIM_1 blockDim.y
#define LDIM_2 blockDim.z

#define GDIM_0 gridDim.x
#define GDIM_1 gridDim.y
#define GDIM_2 gridDim.z
"""




