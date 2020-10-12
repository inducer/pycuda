#!python 
# Conway's Game of Life Accelerated with PyCUDA
# Luis Villasenor
# lvillasen@gmail.com
# 3/26/2016
# Licence: GPLv3
# Usage: python GameOfLife.py n n_iter
# where n is the board size and n_iter the number of iterations
import pycuda.driver as cuda
import pycuda.tools
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import sys
import numpy as np
from pylab import cm as cm
import matplotlib.pyplot as plt
n=int(sys.argv[1])
n_iter=int(sys.argv[2])
n_block=16
n_grid=int(n/n_block);
n=n_block*n_grid;
def random_init(n):
    #np.random.seed(100)
    M=np.zeros((n,n)).astype(np.int32)
    for i in range(n):
        for j in range(n):
            M[j,i]=np.int32(np.random.randint(2))
    return M
mod = SourceModule("""
__global__ void step(int *C, int *M)
{
  int count;
  int n_x = blockDim.x*gridDim.x;
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int threadId = j*n_x+i;
  int i_left; int i_right; int j_down; int j_up;
  if(i==0) {i_left=n_x-1;} else {i_left=i-1;}
  if(i==n_x-1) {i_right=0;} else {i_right=i+1;}
  if(j==0) {j_down=n_x-1;} else {j_down=j-1;}
  if(j==n_x-1) {j_up=0;} else {j_up=j+1;}
  count = C[j*n_x+i_left] + C[j_down*n_x+i]
    + C[j*n_x+i_right] + C[j_up*n_x+i] + C[j_up*n_x+i_left]
    + C[j_down*n_x+i_right] + C[j_down*n_x+i_left]
    + C[j_up*n_x+i_right];

// Modify matrix M according to the rules B3/S23:
//A cell is "Born" if it has exactly 3 neighbours,
//A cell "Survives" if it has 2 or 3 living neighbours; it dies otherwise.
  if(count < 2 || count > 3) M[threadId] = 0; // cell dies
  if(count == 2) M[threadId] = C[threadId];// cell stays the same
  if(count == 3) M[threadId] = 1; // cell either stays alive, or is born
}
""")
func = mod.get_function("step")
C=random_init(n)
M = np.empty_like(C)
C_gpu = gpuarray.to_gpu( C )
M_gpu = gpuarray.to_gpu( M )
for k in range(n_iter):
  func(C_gpu,M_gpu,block=(n_block,n_block,1),grid=(n_grid,n_grid,1))
  C_gpu, M_gpu = M_gpu, C_gpu
print(("%d live cells after %d iterations" %(np.sum(C_gpu.get()),n_iter)))
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111)
fig.suptitle("Conway's Game of Life Accelerated with PyCUDA")
ax.set_title('Number of Iterations = %d'%(n_iter))
myobj =plt.imshow(C_gpu.get(),origin='lower',cmap='Greys',  interpolation='nearest',vmin=0, vmax=1)
plt.pause(.01)
plt.draw()
m=n_iter
while True:
    m+=1
    func(C_gpu,M_gpu,block=(n_block,n_block,1),grid=(n_grid,n_grid,1))
    C_gpu, M_gpu = M_gpu, C_gpu
    myobj.set_data(C_gpu.get())
    ax.set_title('Number of Iterations = %d'%(m))
    plt.pause(.01)
    plt.draw()

