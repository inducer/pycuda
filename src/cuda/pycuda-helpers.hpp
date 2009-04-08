#ifndef _AFJKDASLFSADHF_HEADER_SEEN_PYCUDA_HELPERS_HPP
#define _AFJKDASLFSADHF_HEADER_SEEN_PYCUDA_HELPERS_HPP

extern "C++" {
  // "double-precision" textures ------------------------------------------------
  /* Thanks to Nathan Bell <nbell@nvidia.com> for help in figuring this out. */

  typedef float fp_tex_float;
  typedef int2 fp_tex_double;

  template <enum cudaTextureReadMode read_mode>
  __device__ float fp_tex1Dfetch(texture<float, 1, read_mode> tex, int i)
  {
    return tex1Dfetch(tex, i);
  }

  template <enum cudaTextureReadMode read_mode>
  __device__ double fp_tex1Dfetch(texture<fp_tex_double, 1, read_mode> tex, int i)
  {
    fp_tex_double v = tex1Dfetch(tex, i);
    return __hiloint2double(v.y, v.x);
  }

  template <enum cudaTextureReadMode read_mode>
  __device__ float fp_tex2D(texture<float, 2, read_mode> tex, int i, int j)
  {
    return tex2D(tex, i, j);
  }

  template <enum cudaTextureReadMode read_mode>
  __device__ double fp_tex2D(texture<fp_tex_double, 2, read_mode> tex, int i, int j)
  {
    fp_tex_double v = tex2D(tex, i, j);
    return __hiloint2double(v.y, v.x);
  }

  template <enum cudaTextureReadMode read_mode>
  __device__ float fp_tex3D(texture<float, 3, read_mode> tex, int i, int j, int k)
  {
    return tex3D(tex, i, j, k);
  }

  template <enum cudaTextureReadMode read_mode>
  __device__ double fp_tex3D(texture<fp_tex_double, 2, read_mode> tex, int i, int j, int k)
  {
    fp_tex_double v = tex3D(tex, i, j, k);
    return __hiloint2double(v.y, v.x);
  }
}

#endif
