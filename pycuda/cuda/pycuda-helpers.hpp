#include <pycuda-complex.hpp>
#include <surface_functions.h>
#ifndef _AFJKDASLFSADHF_HEADER_SEEN_PYCUDA_HELPERS_HPP
#define _AFJKDASLFSADHF_HEADER_SEEN_PYCUDA_HELPERS_HPP

extern "C++" {
  // "double-precision" textures ------------------------------------------------
  /* Thanks to Nathan Bell <nbell@nvidia.com> for help in figuring this out. */

  typedef float fp_tex_float;
  typedef int2 fp_tex_double;
  typedef uint2 fp_tex_cfloat;
  typedef int4 fp_tex_cdouble;

   template <enum cudaTextureReadMode read_mode>
  __device__ pycuda::complex<float> fp_tex1Dfetch(texture<fp_tex_cfloat, 1, read_mode> tex, int i)
  {
    fp_tex_cfloat v = tex1Dfetch(tex, i);
    pycuda::complex<float> out;
    return pycuda::complex<float>(__int_as_float(v.x), __int_as_float(v.y));
  }

  template <enum cudaTextureReadMode read_mode>
  __device__ pycuda::complex<double> fp_tex1Dfetch(texture<fp_tex_cdouble, 1, read_mode> tex, int i)
  {
    fp_tex_cdouble v = tex1Dfetch(tex, i);
    return pycuda::complex<double>(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
  }

  template <enum cudaTextureReadMode read_mode>
  __device__ double fp_tex1Dfetch(texture<fp_tex_double, 1, read_mode> tex, int i)
  {
    fp_tex_double v = tex1Dfetch(tex, i);
    return __hiloint2double(v.y, v.x);
  }

// 2D functionality

  template <enum cudaTextureReadMode read_mode>
  __device__ double fp_tex2D(texture<fp_tex_double, 2, read_mode> tex, int i, int j)
  {
    fp_tex_double v = tex2D(tex, i, j);
    return __hiloint2double(v.y, v.x);
  }

  template <enum cudaTextureReadMode read_mode>
  __device__ pycuda::complex<float> fp_tex2D(texture<fp_tex_cfloat, 2, read_mode> tex, int i, int j)
  {
    fp_tex_cfloat v = tex2D(tex, i, j);
    return pycuda::complex<float>(__int_as_float(v.x), __int_as_float(v.y));
  }

  template <enum cudaTextureReadMode read_mode>
  __device__ pycuda::complex<double> fp_tex2D(texture<fp_tex_cdouble, 2, read_mode> tex, int i, int j)
  {
    fp_tex_cdouble v = tex2D(tex, i, j);
    return pycuda::complex<double>(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
  }
  // 2D Layered extension

  template <enum cudaTextureReadMode read_mode>
  __device__ double fp_tex2DLayered(texture<fp_tex_double, cudaTextureType2DLayered, read_mode> tex, float i, float j, int layer)
  {
    fp_tex_double v = tex2DLayered(tex, i, j, layer);
    return __hiloint2double(v.y, v.x);
  }

  template <enum cudaTextureReadMode read_mode>
  __device__ pycuda::complex<float> fp_tex2DLayered(texture<fp_tex_cfloat, cudaTextureType2DLayered, read_mode> tex, float i, float j, int layer)
  {
    fp_tex_cfloat v = tex2DLayered(tex, i, j, layer);
    return pycuda::complex<float>(__int_as_float(v.x), __int_as_float(v.y));
  }

  template <enum cudaTextureReadMode read_mode>
  __device__ pycuda::complex<double> fp_tex2DLayered(texture<fp_tex_cdouble, cudaTextureType2DLayered, read_mode> tex, float i, float j, int layer)
  {
    fp_tex_cdouble v = tex2DLayered(tex, i, j, layer);
    return pycuda::complex<double>(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
  }

  // 3D functionality

  template <enum cudaTextureReadMode read_mode>
  __device__ double fp_tex3D(texture<fp_tex_double, 3, read_mode> tex, int i, int j, int k)
  {
    fp_tex_double v = tex3D(tex, i, j, k);
    return __hiloint2double(v.y, v.x);
  }

  template <enum cudaTextureReadMode read_mode>
  __device__ pycuda::complex<float> fp_tex3D(texture<fp_tex_cfloat, 3, read_mode> tex, int i, int j, int k)
  {
    fp_tex_cfloat v = tex3D(tex, i, j, k);
    return pycuda::complex<float>(__int_as_float(v.x), __int_as_float(v.y));
  }

  template <enum cudaTextureReadMode read_mode>
  __device__ pycuda::complex<double> fp_tex3D(texture<fp_tex_cdouble, 3, read_mode> tex, int i, int j, int k)
  {
    fp_tex_cdouble v = tex3D(tex, i, j, k);
    return pycuda::complex<double>(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
  }

  // FP_Surfaces with complex supprt

  __device__ void fp_surf2DLayeredwrite(double var,surface<void, cudaSurfaceType2DLayered> surf, int i, int j, int layer, enum cudaSurfaceBoundaryMode mode)
  {
    fp_tex_double auxvar;
    auxvar.x =  __double2loint(var);
    auxvar.y =  __double2hiint(var);
    surf2DLayeredwrite(auxvar, surf, i*sizeof(fp_tex_double), j, layer, mode);
  }

  __device__ void fp_surf2DLayeredwrite(pycuda::complex<float> var,surface<void, cudaSurfaceType2DLayered> surf, int i, int j, int layer, enum cudaSurfaceBoundaryMode mode)
  {
    fp_tex_cfloat auxvar;
    auxvar.x =  __float_as_int(var._M_re);
    auxvar.y =  __float_as_int(var._M_im);
    surf2DLayeredwrite(auxvar, surf, i*sizeof(fp_tex_cfloat), j, layer,mode);
  }

  __device__ void fp_surf2DLayeredwrite(pycuda::complex<double> var,surface<void, cudaSurfaceType2DLayered> surf, int i, int j, int layer, enum cudaSurfaceBoundaryMode mode)
  {
    fp_tex_cdouble auxvar;
    auxvar.x =  __double2loint(var._M_re);
    auxvar.y =  __double2hiint(var._M_re);

    auxvar.z = __double2loint(var._M_im);
    auxvar.w = __double2hiint(var._M_im);
    surf2DLayeredwrite(auxvar, surf, i*sizeof(fp_tex_cdouble), j, layer,mode);
  }

  __device__ void fp_surf3Dwrite(double var,surface<void, 3> surf, int i, int j, int k, enum cudaSurfaceBoundaryMode mode)
  {
    fp_tex_double auxvar;
    auxvar.x =  __double2loint(var);
    auxvar.y =  __double2hiint(var);
    surf3Dwrite(auxvar, surf, i*sizeof(fp_tex_double), j, k,mode);
  }

  __device__ void fp_surf3Dwrite(pycuda::complex<float> var,surface<void, 3> surf, int i, int j, int k, enum cudaSurfaceBoundaryMode mode)
  {
    fp_tex_cfloat auxvar;
    auxvar.x =  __float_as_int(var._M_re);
    auxvar.y =  __float_as_int(var._M_im);

    surf3Dwrite(auxvar, surf, i*sizeof(fp_tex_cfloat), j, k, mode);
  }

  __device__ void fp_surf3Dwrite(pycuda::complex<double> var,surface<void, 3> surf, int i, int j, int k, enum cudaSurfaceBoundaryMode mode)
  {
    fp_tex_cdouble auxvar;
    auxvar.x =  __double2loint(var._M_re);
    auxvar.y =  __double2hiint(var._M_re);

    auxvar.z = __double2loint(var._M_im);
    auxvar.w = __double2hiint(var._M_im);
    surf3Dwrite(auxvar, surf, i*sizeof(fp_tex_cdouble), j, k, mode);
  }

  __device__ void fp_surf2DLayeredread(double *var, surface<void, cudaSurfaceType2DLayered> surf, int i, int j, int layer, enum cudaSurfaceBoundaryMode mode)
  {
    fp_tex_double v;
    surf2DLayeredread(&v, surf, i*sizeof(fp_tex_double), j, layer, mode);
    *var = __hiloint2double(v.y, v.x);
  }

  __device__ void fp_surf2DLayeredread(pycuda::complex<float> *var, surface<void, cudaSurfaceType2DLayered> surf, int i, int j, int layer, enum cudaSurfaceBoundaryMode mode)
  {
    fp_tex_cfloat v;
    surf2DLayeredread(&v, surf, i*sizeof(fp_tex_cfloat), j, layer, mode);
    *var = pycuda::complex<float>(__int_as_float(v.x), __int_as_float(v.y));
  }

  __device__ void fp_surf2DLayeredread(pycuda::complex<double> *var, surface<void, cudaSurfaceType2DLayered> surf, int i, int j, int layer, enum cudaSurfaceBoundaryMode mode)
  {
    fp_tex_cdouble v;
    surf2DLayeredread(&v, surf, i*sizeof(fp_tex_cdouble), j, layer, mode);
    *var = pycuda::complex<double>(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
  }

  __device__ void fp_surf3Dread(double *var, surface<void, 3> surf, int i, int j, int k, enum cudaSurfaceBoundaryMode mode)
  {
    fp_tex_double v;
    surf3Dread(&v, surf, i*sizeof(fp_tex_double), j, k, mode);
    *var = __hiloint2double(v.y, v.x);
  }

  __device__ void fp_surf3Dread(pycuda::complex<float> *var, surface<void, 3> surf, int i, int j, int k, enum cudaSurfaceBoundaryMode mode)
  {
    fp_tex_cfloat v;
    surf3Dread(&v, surf, i*sizeof(fp_tex_cfloat), j, k, mode);
    *var = pycuda::complex<float>(__int_as_float(v.x), __int_as_float(v.y));
  }

  __device__ void fp_surf3Dread(pycuda::complex<double> *var, surface<void, 3> surf, int i, int j, int k, enum cudaSurfaceBoundaryMode mode)
  {
    fp_tex_cdouble v;
    surf3Dread(&v, surf, i*sizeof(fp_tex_cdouble), j, k, mode);
    *var = pycuda::complex<double>(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
  }
#define PYCUDA_GENERATE_FP_TEX_FUNCS(TYPE) \
  template <enum cudaTextureReadMode read_mode> \
  __device__ TYPE fp_tex1Dfetch(texture<TYPE, 1, read_mode> tex, int i) \
  { \
    return tex1Dfetch(tex, i); \
  } \
  template <enum cudaTextureReadMode read_mode> \
  __device__ TYPE fp_tex2D(texture<TYPE, 2, read_mode> tex, int i, int j) \
  { \
    return tex2D(tex, i, j); \
  } \
  template <enum cudaTextureReadMode read_mode> \
   __device__ TYPE fp_tex2DLayered(texture<TYPE, cudaTextureType2DLayered, read_mode> tex, int i, int j, int layer) \
  { \
    return tex2DLayered(tex, i, j, layer); \
  } \
  template <enum cudaTextureReadMode read_mode> \
  __device__ TYPE fp_tex3D(texture<TYPE, 3, read_mode> tex, int i, int j, int k) \
  { \
    return tex3D(tex, i, j, k); \
  } \
  __device__ void fp_surf2DLayeredwrite(TYPE var,surface<void, cudaSurfaceType2DLayered> surf, int i, int j, int layer,enum cudaSurfaceBoundaryMode mode) \
  { \
    surf2DLayeredwrite(var, surf, i*sizeof(TYPE), j, layer, mode); \
  } \
  __device__ void fp_surf2DLayeredread(TYPE *var, surface<void, cudaSurfaceType2DLayered> surf, int i, int j, int layer,enum cudaSurfaceBoundaryMode mode) \
  { \
    surf2DLayeredread(var, surf, i*sizeof(TYPE), j, layer, mode); \
  } \
  __device__ void fp_surf3Dwrite(TYPE var,surface<void, 3> surf, int i, int j, int k, enum cudaSurfaceBoundaryMode mode) \
  { \
    surf3Dwrite(var, surf, i*sizeof(TYPE), j, k, mode); \
  } \
  __device__ void fp_surf3Dread(TYPE *var, surface<void, 3> surf, int i, int j, int k, enum cudaSurfaceBoundaryMode mode) \
  { \
    surf3Dread(var, surf, i*sizeof(TYPE), j, k, mode); \
  }
  PYCUDA_GENERATE_FP_TEX_FUNCS(float)
  PYCUDA_GENERATE_FP_TEX_FUNCS(int)
  PYCUDA_GENERATE_FP_TEX_FUNCS(unsigned int)
  PYCUDA_GENERATE_FP_TEX_FUNCS(short int)
  PYCUDA_GENERATE_FP_TEX_FUNCS(unsigned short int)
  PYCUDA_GENERATE_FP_TEX_FUNCS(char)
  PYCUDA_GENERATE_FP_TEX_FUNCS(unsigned char)
}

#endif
