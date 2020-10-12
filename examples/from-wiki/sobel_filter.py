#!python 
#!/usr/bin/env python
#-*- coding: utf-8 -*-
#
# Requires PyCuda, PyOpenGL, and Pil
# MAKE SURE YOU HAVE AN UPDATED VERSION OF THESE PACKAGES!!
#
# Ported to PyCUDA by
# Stefano Brilli: stefanobrilli@gmail.com
#
# Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
#
# This software contains source code provided by NVIDIA Corporation
#
# http://developer.download.nvidia.com/compute/cuda/2_3/sdk/docs/cudasdk_eula.pdf
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
#

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL.ARB.vertex_buffer_object import *
import numpy as np, Image
import sys, time, os
import pycuda.driver as cuda_driver
import pycuda.gl as cuda_gl
import pycuda
#import pycuda.gl.autoinit
from pycuda.compiler import SourceModule

imWidth = 0
imHeight = 0
wWidth = 0
wHeight = 0
wName = "Cuda Edge Detection:"
pixels = None
array = None
texid = 0
pbo_buffer = None
cuda_pbo_resource = None
mode = 0
scale = 1.0

frameCount = 0
fpsCount = 0
fpsLimit = 8
timer = 0.0
ver2011 = False

def copy2D_array_to_device(dst, src, type_sz, width, height):
    copy = cuda_driver.Memcpy2D()
    copy.set_src_array(src)
    copy.set_dst_device(dst)
    copy.height = height
    copy.dst_pitch = copy.src_pitch = copy.width_in_bytes = width*type_sz
    copy(aligned=True)

def computeFPS():
    global frameCount, fpsCount, fpsLimit, timer
    frameCount += 1
    fpsCount += 1
    if fpsCount == fpsLimit:
        ifps = 1.0 /timer
        glutSetWindowTitle("Cuda Edge Detection: %f fps" % ifps)
        fpsCount = 0

def sobelFilter(odata, iw, ih):
    global array, pixels, mode, scale
    if mode == 3:
        # Texture and shared memory with fixed BlockSize
        sm = SourceModule("""
            texture<unsigned char, 2> tex;
            extern __shared__ unsigned char LocalBlock[];
            #define RADIUS 1
            #define BlockWidth 80
            #define SharedPitch 384
            __device__ unsigned char
            ComputeSobel(unsigned char ul, // upper left
                         unsigned char um, // upper middle
                         unsigned char ur, // upper right
                         unsigned char ml, // middle left
                         unsigned char mm, // middle (unused)
                         unsigned char mr, // middle right
                         unsigned char ll, // lower left
                         unsigned char lm, // lower middle
                         unsigned char lr, // lower right
                         float fScale )
            {
                short Horz = ur + 2*mr + lr - ul - 2*ml - ll;
                short Vert = ul + 2*um + ur - ll - 2*lm - lr;
                short Sum = (short) (fScale*(::abs(int(Horz))+::abs(int(Vert))));
                if ( Sum < 0 ) return 0; else if ( Sum > 0xff ) return 0xff;
                return (unsigned char) Sum;
            }

            __global__ void
            SobelShared( int* pSobelOriginal, unsigned short SobelPitch,
                         short w, short h, float fScale )
            {
                short u = 4*blockIdx.x*BlockWidth;
                short v = blockIdx.y*blockDim.y + threadIdx.y;
                short ib;

                int SharedIdx = threadIdx.y * SharedPitch;

                for ( ib = threadIdx.x; ib < BlockWidth+2*RADIUS; ib += blockDim.x ) {
                    LocalBlock[SharedIdx+4*ib+0] = tex2D( tex,
                        (float) (u+4*ib-RADIUS+0), (float) (v-RADIUS) );
                    LocalBlock[SharedIdx+4*ib+1] = tex2D( tex,
                        (float) (u+4*ib-RADIUS+1), (float) (v-RADIUS) );
                    LocalBlock[SharedIdx+4*ib+2] = tex2D( tex,
                        (float) (u+4*ib-RADIUS+2), (float) (v-RADIUS) );
                    LocalBlock[SharedIdx+4*ib+3] = tex2D( tex,
                        (float) (u+4*ib-RADIUS+3), (float) (v-RADIUS) );
                }
                if ( threadIdx.y < RADIUS*2 ) {
                    //
                    // copy trailing RADIUS*2 rows of pixels into shared
                    //
                    SharedIdx = (blockDim.y+threadIdx.y) * SharedPitch;
                    for ( ib = threadIdx.x; ib < BlockWidth+2*RADIUS; ib += blockDim.x ) {
                        LocalBlock[SharedIdx+4*ib+0] = tex2D( tex,
                            (float) (u+4*ib-RADIUS+0), (float) (v+blockDim.y-RADIUS) );
                        LocalBlock[SharedIdx+4*ib+1] = tex2D( tex,
                            (float) (u+4*ib-RADIUS+1), (float) (v+blockDim.y-RADIUS) );
                        LocalBlock[SharedIdx+4*ib+2] = tex2D( tex,
                            (float) (u+4*ib-RADIUS+2), (float) (v+blockDim.y-RADIUS) );
                        LocalBlock[SharedIdx+4*ib+3] = tex2D( tex,
                            (float) (u+4*ib-RADIUS+3), (float) (v+blockDim.y-RADIUS) );
                    }
                }

                __syncthreads();

                u >>= 2;    // index as uchar4 from here
                uchar4 *pSobel = (uchar4 *) (((char *) pSobelOriginal)+v*SobelPitch);
                SharedIdx = threadIdx.y * SharedPitch;

                for ( ib = threadIdx.x; ib < BlockWidth; ib += blockDim.x ) {

                    unsigned char pix00 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+0];
                    unsigned char pix01 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+1];
                    unsigned char pix02 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+2];
                    unsigned char pix10 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+0];
                    unsigned char pix11 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+1];
                    unsigned char pix12 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+2];
                    unsigned char pix20 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+0];
                    unsigned char pix21 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+1];
                    unsigned char pix22 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+2];

                    uchar4 out;

                    out.x = ComputeSobel(pix00, pix01, pix02,
                                         pix10, pix11, pix12,
                                         pix20, pix21, pix22, fScale );

                    pix00 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+3];
                    pix10 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+3];
                    pix20 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+3];
                    out.y = ComputeSobel(pix01, pix02, pix00,
                                         pix11, pix12, pix10,
                                         pix21, pix22, pix20, fScale );

                    pix01 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+4];
                    pix11 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+4];
                    pix21 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+4];
                    out.z = ComputeSobel( pix02, pix00, pix01,
                                          pix12, pix10, pix11,
                                          pix22, pix20, pix21, fScale );

                    pix02 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+5];
                    pix12 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+5];
                    pix22 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+5];
                    out.w = ComputeSobel( pix00, pix01, pix02,
                                          pix10, pix11, pix12,
                                          pix20, pix21, pix22, fScale );
                    if ( u+ib < w/4 && v < h ) {
                        pSobel[u+ib] = out;
                    }
                }

                __syncthreads();
            }
        """)
        cuda_function = sm.get_function("SobelShared")
    elif mode == 2:
        # Texture and shared memory with variable BlockSize
        sm = SourceModule("""
        #define RADIUS 1
        texture<unsigned char, 2> tex;
        extern __shared__ unsigned char LocalBlock[];
        __device__ unsigned char
        ComputeSobel(unsigned char ul, // upper left
                     unsigned char um, // upper middle
                     unsigned char ur, // upper right
                     unsigned char ml, // middle left
                     unsigned char mm, // middle (unused)
                     unsigned char mr, // middle right
                     unsigned char ll, // lower left
                     unsigned char lm, // lower middle
                     unsigned char lr, // lower right
                     float fScale )
        {
            short Horz = ur + 2*mr + lr - ul - 2*ml - ll;
            short Vert = ul + 2*um + ur - ll - 2*lm - lr;
            short Sum = (short) (fScale*(::abs(int(Horz))+::abs(int(Vert))));
            if ( Sum < 0 ) return 0; else if ( Sum > 0xff ) return 0xff;
            return (unsigned char) Sum;
        }

        __global__ void
        SobelShared( int* pSobelOriginal, unsigned short SobelPitch,
                     short BlockWidth, short SharedPitch,
                     short w, short h, float fScale )
        {
            short u = 4*blockIdx.x*BlockWidth;
            short v = blockIdx.y*blockDim.y + threadIdx.y;
            short ib;

            int SharedIdx = threadIdx.y * SharedPitch;

            for ( ib = threadIdx.x; ib < BlockWidth+2*RADIUS; ib += blockDim.x ) {
                LocalBlock[SharedIdx+4*ib+0] = tex2D( tex,
                    (float) (u+4*ib-RADIUS+0), (float) (v-RADIUS) );
                LocalBlock[SharedIdx+4*ib+1] = tex2D( tex,
                    (float) (u+4*ib-RADIUS+1), (float) (v-RADIUS) );
                LocalBlock[SharedIdx+4*ib+2] = tex2D( tex,
                    (float) (u+4*ib-RADIUS+2), (float) (v-RADIUS) );
                LocalBlock[SharedIdx+4*ib+3] = tex2D( tex,
                    (float) (u+4*ib-RADIUS+3), (float) (v-RADIUS) );
            }
            if ( threadIdx.y < RADIUS*2 ) {
                //
                // copy trailing RADIUS*2 rows of pixels into shared
                //
                SharedIdx = (blockDim.y+threadIdx.y) * SharedPitch;
                for ( ib = threadIdx.x; ib < BlockWidth+2*RADIUS; ib += blockDim.x ) {
                    LocalBlock[SharedIdx+4*ib+0] = tex2D( tex,
                        (float) (u+4*ib-RADIUS+0), (float) (v+blockDim.y-RADIUS) );
                    LocalBlock[SharedIdx+4*ib+1] = tex2D( tex,
                        (float) (u+4*ib-RADIUS+1), (float) (v+blockDim.y-RADIUS) );
                    LocalBlock[SharedIdx+4*ib+2] = tex2D( tex,
                        (float) (u+4*ib-RADIUS+2), (float) (v+blockDim.y-RADIUS) );
                    LocalBlock[SharedIdx+4*ib+3] = tex2D( tex,
                        (float) (u+4*ib-RADIUS+3), (float) (v+blockDim.y-RADIUS) );
                }
            }

            __syncthreads();

            u >>= 2;    // index as uchar4 from here
            uchar4 *pSobel = (uchar4 *) (((char *) pSobelOriginal)+v*SobelPitch);
            SharedIdx = threadIdx.y * SharedPitch;

            for ( ib = threadIdx.x; ib < BlockWidth; ib += blockDim.x ) {

                unsigned char pix00 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+0];
                unsigned char pix01 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+1];
                unsigned char pix02 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+2];
                unsigned char pix10 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+0];
                unsigned char pix11 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+1];
                unsigned char pix12 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+2];
                unsigned char pix20 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+0];
                unsigned char pix21 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+1];
                unsigned char pix22 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+2];

                uchar4 out;

                out.x = ComputeSobel(pix00, pix01, pix02,
                                     pix10, pix11, pix12,
                                     pix20, pix21, pix22, fScale );

                pix00 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+3];
                pix10 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+3];
                pix20 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+3];
                out.y = ComputeSobel(pix01, pix02, pix00,
                                     pix11, pix12, pix10,
                                     pix21, pix22, pix20, fScale );

                pix01 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+4];
                pix11 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+4];
                pix21 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+4];
                out.z = ComputeSobel( pix02, pix00, pix01,
                                      pix12, pix10, pix11,
                                      pix22, pix20, pix21, fScale );

                pix02 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+5];
                pix12 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+5];
                pix22 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+5];
                out.w = ComputeSobel( pix00, pix01, pix02,
                                      pix10, pix11, pix12,
                                      pix20, pix21, pix22, fScale );
                if ( u+ib < w/4 && v < h ) {
                    pSobel[u+ib] = out;
                }
            }

            __syncthreads();
        }

        """)
        cuda_function = sm.get_function("SobelShared")
    if mode == 1:
        # Just Texture
        sm = SourceModule("""
        texture<unsigned char, 2> tex;
        __device__ unsigned char ComputeSobel(unsigned char ul, // upper left
                     unsigned char um, // upper middle
                     unsigned char ur, // upper right
                     unsigned char ml, // middle left
                     unsigned char mm, // middle (unused)
                     unsigned char mr, // middle right
                     unsigned char ll, // lower left
                     unsigned char lm, // lower middle
                     unsigned char lr, // lower right
                     float fScale )
        {
            short Horz = ur + 2*mr + lr - ul - 2*ml - ll;
            short Vert = ul + 2*um + ur - ll - 2*lm - lr;
            short Sum = (short) (fScale*(::abs(int(Horz))+::abs(int(Vert))));
            if ( Sum < 0 ) return 0; else if ( Sum > 0xff ) return 0xff;
            return (unsigned char) Sum;
        }
        __global__ void SobelTex( int* pSobelOriginal, unsigned int Pitch,
                  int w, int h, float fScale )
        {
            unsigned char *pSobel =
              (unsigned char *) (((char *) pSobelOriginal)+blockIdx.x*Pitch);
            for ( int i = threadIdx.x; i < w; i += blockDim.x ) {
                unsigned char pix00 = tex2D( tex, (float) i-1, (float) blockIdx.x-1 );
                unsigned char pix01 = tex2D( tex, (float) i+0, (float) blockIdx.x-1 );
                unsigned char pix02 = tex2D( tex, (float) i+1, (float) blockIdx.x-1 );
                unsigned char pix10 = tex2D( tex, (float) i-1, (float) blockIdx.x+0 );
                unsigned char pix11 = tex2D( tex, (float) i+0, (float) blockIdx.x+0 );
                unsigned char pix12 = tex2D( tex, (float) i+1, (float) blockIdx.x+0 );
                unsigned char pix20 = tex2D( tex, (float) i-1, (float) blockIdx.x+1 );
                unsigned char pix21 = tex2D( tex, (float) i+0, (float) blockIdx.x+1 );
                unsigned char pix22 = tex2D( tex, (float) i+1, (float) blockIdx.x+1 );
                pSobel[i] = ComputeSobel(pix00, pix01, pix02,
                                         pix10, pix11, pix12,
                                         pix20, pix21, pix22, fScale );
            }
        }
        """)
        cuda_function = sm.get_function("SobelTex")
    elif mode == 0:
        # Just Copy
        sm = SourceModule("""
        texture<unsigned char, 2> tex;
        __global__ void SobelCopyImage(int* pSobelOriginal, unsigned int Pitch, int w, int h, float fscale )
        {
            unsigned char *pSobel =
              (unsigned char *) (((unsigned char *) pSobelOriginal)+blockIdx.x*Pitch);
            for ( int i = threadIdx.x; i < w; i += blockDim.x ) {
                pSobel[i] = min( max((tex2D( tex, (float)i, (float)blockIdx.x ) * fscale), 0.f), 255.f);
            }
        }
        """)
        cuda_function = sm.get_function("SobelCopyImage")
    texref = sm.get_texref("tex")
    texref.set_array(array)
    texref.set_flags(cuda_driver.TRSA_OVERRIDE_FORMAT)
    if mode == 3:
        # fixed BlockSize Launch
        RADIUS = 1
        threads = (16, 4, 1)
        BlockWidth = 80 # Do not change!
        blocks = (iw/(4*BlockWidth)+(0!=iw%(4*BlockWidth)),
                               ih/threads[1]+(0!=ih%threads[1]) )
        SharedPitch = ~0x3f & (4*(BlockWidth+2*RADIUS)+0x3f);
        sharedMem = SharedPitch*(threads[1]+2*RADIUS);
        iw = iw & ~3
        cuda_function(np.intp(odata), np.uint16(iw), np.int16(iw), np.int16(ih), np.float32(scale), texrefs=[texref],block=threads, grid=blocks, shared=sharedMem)
    elif mode == 2:
        # variable BlockSize launch
        RADIUS = 1
        threads = (16, 4, 1)
        BlockWidth = 80 # Change only with divisible by 16 values!
        blocks = (iw/(4*BlockWidth)+(0!=iw%(4*BlockWidth)),
                               ih/threads[1]+(0!=ih%threads[1]) )
        SharedPitch = ~0x3f & (4*(BlockWidth+2*RADIUS)+0x3f);
        sharedMem = SharedPitch*(threads[1]+2*RADIUS);
        iw = iw & ~3
        cuda_function(np.intp(odata), np.uint16(iw), np.int16(BlockWidth), np.int16(SharedPitch), np.int16(iw), np.int16(ih), np.float32(scale), texrefs=[texref],block=threads, grid=blocks, shared=sharedMem)
    else:
        BlockWidth = 384
        cuda_function(np.intp(odata), np.uint32(iw), np.int32(iw), np.int32(ih), np.float32(scale), texrefs=[texref],block=(BlockWidth,1,1),grid=(ih,1))

def initGL():
    global wWidth, wHeight, wName
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA)
    glutInitWindowSize(wWidth, wHeight)
    glutCreateWindow(wName)
    import pycuda.gl.autoinit

def loadImage(fn=None):
    global pixels, imWidth, imHeight, wWidth, wHeight
    try:
        im = Image.open(fn) # Open the image
    except IOError:
        print("Usage:", os.path.basename(sys.argv[0]), "[IMAGE=defaultimage.jpg]")
        print("Can't open", fn)
        sys.exit(1)
    imWidth, imHeight = im.size # Window size is set to image size
    wWidth, wHeight = im.size
    im.draft("L", im.size) # L-flag is for Luminance
    pixels = np.fromstring(im.tostring(), dtype=np.uint8) # Got the array
    pixels.resize((imHeight, imWidth)) # Resize to 2d array
    print("Reading image:", fn, "size:", imWidth, "x", imHeight)

def initData(fn=None):
    global pixels, array, pbo_buffer, cuda_pbo_resource, imWidth, imHeight, texid

    # Cuda array initialization
    array = cuda_driver.matrix_to_array(pixels, "C") # C-style instead of Fortran-style: row-major

    pixels.fill(0) # Resetting the array to 0

    pbo_buffer = glGenBuffers(1) # generate 1 buffer reference
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer) # binding to this buffer
    glBufferData(GL_PIXEL_UNPACK_BUFFER, imWidth*imHeight, pixels, GL_STREAM_DRAW) # Allocate the buffer
    bsize = glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE) # Check allocated buffer size
    assert(bsize == imWidth*imHeight)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0) # Unbind

    if ver2011:
        cuda_pbo_resource = pycuda.gl.RegisteredBuffer(int(pbo_buffer), cuda_gl.graphics_map_flags.WRITE_DISCARD)
    else:
        cuda_pbo_resource = cuda_gl.BufferObject(int(pbo_buffer)) # Mapping GLBuffer to cuda_resource


    glGenTextures(1, texid); # generate 1 texture reference
    glBindTexture(GL_TEXTURE_2D, texid); # binding to this texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, imWidth, imHeight,  0, GL_LUMINANCE, GL_UNSIGNED_BYTE, None); # Allocate the texture
    glBindTexture(GL_TEXTURE_2D, 0) # Unbind

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1) # 1-byte row alignment
    glPixelStorei(GL_PACK_ALIGNMENT, 1) # 1-byte row alignment


def display():
    global cuda_pbo_resource, pbo_buffer, texid, imWidth, imHeight, timer

    timer = time.time() # Starting timer
    mapping_obj = cuda_pbo_resource.map() # Map the GlBuffer
    if ver2011:
        data, sz = mapping_obj.device_ptr_and_size() # Got the CUDA pointer to GlBuffer
    else:
        data = mapping_obj.device_ptr()
    sobelFilter(data, imWidth, imHeight) # Writing to "data"
    mapping_obj.unmap() # Unmap the GlBuffer

    glClear(GL_COLOR_BUFFER_BIT) # Clear
    glBindTexture(GL_TEXTURE_2D, texid)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer)
    # Copyng from buffer to texture
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imWidth, imHeight, GL_LUMINANCE, GL_UNSIGNED_BYTE, None)
    #glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, imWidth, imHeight,  0, GL_LUMINANCE, GL_UNSIGNED_BYTE, None);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0) # Unbind

    glDisable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

    glBegin(GL_QUADS)
    glVertex2f(0, 0)
    glTexCoord2f(0, 0)
    glVertex2f(0, 1)
    glTexCoord2f(1, 0)
    glVertex2f(1, 1)
    glTexCoord2f(1, 1)
    glVertex2f(1, 0)
    glTexCoord2f(0, 1)
    glEnd()
    glBindTexture(GL_TEXTURE_2D, 0)
    glutSwapBuffers()
    timer = time.time()-timer
    computeFPS()
    glutPostRedisplay()

def reshape(x, y):
    glViewport(0, 0, x, y)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, 1, 0, 1, 0, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glutPostRedisplay()

def keyboard(key, x=0, y=0):
    global mode, scale
    if key=="q":
        sys.exit(0)
    elif key=="I" or key=="i":
        mode = 0
    elif key=="T" or key=="t":
        mode = 1
    elif key=="S" or key=="s":
        mode = 2
    elif key=="D" or key=="d":
        mode = 3
    elif key == "-":
        scale -= 0.1
    elif key == "=":
        scale += 0.1

def idle():
    glutPostRedisplay()

def main(argv):
    fn = "defaultimage.jpg"
    if len(argv) > 1:
        fn = argv[1]

    loadImage(fn) # Loading the image

    initGL()
    initData(fn)
    print("""
    Q: Exit
    I: display image
    T: display Sobel edge detection (computed with tex)
    S: display Sobel edge detection (computed with tex+shared memory)
    D: display Sobel edge detection (computed with tex+shared memory+fixed block size)
    Use the '-' and '=' keys to change the brightness.

    TESTED WITH IMAGE SIZE OF 512x512... just like the original demo.
    Other image sizes may not work
    """)
    glutDisplayFunc(display)
    glutKeyboardFunc(keyboard)
    glutReshapeFunc(reshape)
    glutIdleFunc(idle)
    glutMainLoop();

if __name__ == "__main__":
    if pycuda.VERSION[0] >= 2011:
        ver2011 = True
    main(sys.argv)


