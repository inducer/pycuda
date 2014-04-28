#include <pycuda-complex.hpp>

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

  template <enum cudaTextureReadMode read_mode>
  __device__ double fp_tex2D(texture<fp_tex_double, 2, read_mode> tex, int i, int j)
  {
    fp_tex_double v = tex2D(tex, i, j);
    return __hiloint2double(v.y, v.x);
  }

  template <enum cudaTextureReadMode read_mode>
  __device__ double fp_tex3D(texture<fp_tex_double, 3, read_mode> tex, int i, int j, int k)
  {
    fp_tex_double v = tex3D(tex, i, j, k);
    return __hiloint2double(v.y, v.x);
  }

#define PYCUDA_GENERATE_FP_TEX_FUNCS(TYPE) \
  template <enum cudaTextureReadMode read_mode> \
  __device__ TYPE fp_tex1Dfetch(texture<TYPE, 1, read_mode> tex, int i) \
  { \
    return tex1Dfetch(tex, i); \
  } \
 \
  template <enum cudaTextureReadMode read_mode> \
  __device__ TYPE fp_tex2D(texture<TYPE, 2, read_mode> tex, int i, int j) \
  { \
    return tex2D(tex, i, j); \
  } \
 \
  template <enum cudaTextureReadMode read_mode> \
  __device__ TYPE fp_tex3D(texture<TYPE, 3, read_mode> tex, int i, int j, int k) \
  { \
    return tex3D(tex, i, j, k); \
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
