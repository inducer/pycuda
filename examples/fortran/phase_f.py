#
#A test calculation to brute force calculate the Born approximation amplitude
#from a set of natom atom positions in r with complex scattering factors in f
#and npixel pixel wavevetors in q. The output amplitude is in a.
#
import numpy as np
import time
import sys
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""
   attributes (global) subroutine phase_factor(q,r,f,a,natom,npixel) bind(c)
   use, intrinsic :: iso_fortran_env
   complex(kind=real32), parameter :: cz=(0.0_real32,0.0_real32)
   integer(kind=int32), value, intent(in) :: natom,npixel
   real(kind=real32), intent(in) :: q(3,*),r(3,*)
   complex(kind=real32), intent(in) :: f(*)
   complex(kind=real32), intent(inout) :: a(*)
   complex(kind=real32) :: anow,ex
   real(kind=real32) :: s,c,qdotr
   integer(kind=int32) :: i,ipix
   ipix=threadIdx%x+blockDim%x*(blockIdx%x-1)
   if (ipix.le.npixel) then
      anow=cz
      do i=1,natom
         qdotr=q(1,ipix)*r(1,i)+q(2,ipix)*r(2,i)+q(3,ipix)*r(3,i)
         s=sin(qdotr)
         c=cos(qdotr)
         ex%re=c
         ex%im=s
         anow=anow+f(i)*ex
      enddo
      a(ipix)=anow
   endif
   end subroutine phase_factor
""",nvcc='nvfortran',no_extern_c=True,options=["-O3", "-fast", "-gpu:fastmath"])

cudafun = mod.get_function("phase_factor")

seed = 17
natom = 10000
npixel = 100000
nthread = 256
print("seed ",seed)
print("natom ",natom)
print("npixel ",npixel)
print("nthread",nthread)
nblock = (npixel-1)//nthread+1
rng = np.random.default_rng(seed)
r = (rng.random(3*natom).reshape((natom,3))-0.5).astype(np.float32)
f2 = rng.random(2*natom).reshape((natom,2))-0.5
f = (f2[:,0]+1j*f2[:,1]).astype(np.complex64)
q = (rng.random(3*npixel).reshape((npixel,3))-0.5).astype(np.float32)

print()
print("Cuda calculation start")
t0 = time.time()
r_gpu = cuda.mem_alloc(r.nbytes)
cuda.memcpy_htod(r_gpu,r)
f_gpu = cuda.mem_alloc(f.nbytes)
cuda.memcpy_htod(f_gpu,f)
q_gpu = cuda.mem_alloc(q.nbytes)
cuda.memcpy_htod(q_gpu,q)
ph_gpu = cuda.mem_alloc(8*npixel)
cudafun(q_gpu,r_gpu,f_gpu,ph_gpu,np.int32(natom),np.int32(npixel)
   ,block=(nthread,1,1),grid=(nblock,1,1))
phfromgpu= np.zeros(npixel,dtype=np.complex64)
cuda.memcpy_dtoh(phfromgpu,ph_gpu)
t1 = time.time()
print("cuda time",t1-t0)
print(phfromgpu[0],phfromgpu[-1])

print()
print("CPU calculation start")
t2 = time.time()
ph = np.zeros((npixel),dtype=np.complex64)
for i in range(npixel):
   rdotq = q[i,0]*r[:,0]+q[i,1]*r[:,1]+q[i,2]*r[:,2]
   ph[i] = np.sum(f*(np.cos(rdotq)+1j*np.sin(rdotq)))
t3 = time.time()

print(ph[0],ph[-1])
print("numpy time",t3-t2)
print("cuda speedup",(t3-t2)/(t1-t0))
print("maximum relative difference python/cuda "
   ,np.max(np.abs(ph-phfromgpu)/abs(ph)))
