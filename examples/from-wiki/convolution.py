#!python 
'''
/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/*
 * This sample implements a separable convolution filter
 * of a 2D signal with a gaussian kernel.
 */

 Ported to pycuda by Andrew Wagner <awagner@illinois.edu>, June 2009.
'''

import numpy
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import string

# Pull out a bunch of stuff that was hard coded as pre-processor directives used
# by both the kernel and calling code.
KERNEL_RADIUS = 8
UNROLL_INNER_LOOP = True
KERNEL_W = 2 * KERNEL_RADIUS + 1
ROW_TILE_W = 128
KERNEL_RADIUS_ALIGNED = 16
COLUMN_TILE_W = 16
COLUMN_TILE_H = 48
template = '''
//24-bit multiplication is faster on G80,
//but we must be sure to multiply integers
//only within [-8M, 8M - 1] range
#define IMUL(a, b) __mul24(a, b)

////////////////////////////////////////////////////////////////////////////////
// Kernel configuration
////////////////////////////////////////////////////////////////////////////////
#define KERNEL_RADIUS $KERNEL_RADIUS
#define KERNEL_W $KERNEL_W
__device__ __constant__ float d_Kernel_rows[KERNEL_W];
__device__ __constant__ float d_Kernel_columns[KERNEL_W];

// Assuming ROW_TILE_W, KERNEL_RADIUS_ALIGNED and dataW
// are multiples of coalescing granularity size,
// all global memory operations are coalesced in convolutionRowGPU()
#define            ROW_TILE_W  $ROW_TILE_W
#define KERNEL_RADIUS_ALIGNED  $KERNEL_RADIUS_ALIGNED

// Assuming COLUMN_TILE_W and dataW are multiples
// of coalescing granularity size, all global memory operations
// are coalesced in convolutionColumnGPU()
#define COLUMN_TILE_W $COLUMN_TILE_W
#define COLUMN_TILE_H $COLUMN_TILE_H

////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowGPU(
    float *d_Result,
    float *d_Data,
    int dataW,
    int dataH
){
    //Data cache
    __shared__ float data[KERNEL_RADIUS + ROW_TILE_W + KERNEL_RADIUS];

    //Current tile and apron limits, relative to row start
    const int         tileStart = IMUL(blockIdx.x, ROW_TILE_W);
    const int           tileEnd = tileStart + ROW_TILE_W - 1;
    const int        apronStart = tileStart - KERNEL_RADIUS;
    const int          apronEnd = tileEnd   + KERNEL_RADIUS;

    //Clamp tile and apron limits by image borders
    const int    tileEndClamped = min(tileEnd, dataW - 1);
    const int apronStartClamped = max(apronStart, 0);
    const int   apronEndClamped = min(apronEnd, dataW - 1);

    //Row start index in d_Data[]
    const int          rowStart = IMUL(blockIdx.y, dataW);

    //Aligned apron start. Assuming dataW and ROW_TILE_W are multiples
    //of half-warp size, rowStart + apronStartAligned is also a
    //multiple of half-warp size, thus having proper alignment
    //for coalesced d_Data[] read.
    const int apronStartAligned = tileStart - KERNEL_RADIUS_ALIGNED;

    const int loadPos = apronStartAligned + threadIdx.x;
    //Set the entire data cache contents
    //Load global memory values, if indices are within the image borders,
    //or initialize with zeroes otherwise
    if(loadPos >= apronStart){
        const int smemPos = loadPos - apronStart;

        data[smemPos] =
            ((loadPos >= apronStartClamped) && (loadPos <= apronEndClamped)) ?
            d_Data[rowStart + loadPos] : 0;
    }

    //Ensure the completness of the loading stage
    //because results, emitted by each thread depend on the data,
    //loaded by another threads
    __syncthreads();
    const int writePos = tileStart + threadIdx.x;
    //Assuming dataW and ROW_TILE_W are multiples of half-warp size,
    //rowStart + tileStart is also a multiple of half-warp size,
    //thus having proper alignment for coalesced d_Result[] write.
    if(writePos <= tileEndClamped){
        const int smemPos = writePos - apronStart;
        float sum = 0;
'''
originalLoop = '''
        for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
            sum += data[smemPos + k] * d_Kernel_rows[KERNEL_RADIUS - k];
'''
unrolledLoop = ''
for k in range(-KERNEL_RADIUS,  KERNEL_RADIUS+1):
    loopTemplate = string.Template(
    'sum += data[smemPos + $k] * d_Kernel_rows[KERNEL_RADIUS - $k];\n')
    unrolledLoop += loopTemplate.substitute(k=k)

#print unrolledLoop
template += unrolledLoop if UNROLL_INNER_LOOP else originalLoop
template += '''
        d_Result[rowStart + writePos] = sum;
        //d_Result[rowStart + writePos] = 128;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnGPU(
    float *d_Result,
    float *d_Data,
    int dataW,
    int dataH,
    int smemStride,
    int gmemStride
){
    //Data cache
    __shared__ float data[COLUMN_TILE_W *
    (KERNEL_RADIUS + COLUMN_TILE_H + KERNEL_RADIUS)];

    //Current tile and apron limits, in rows
    const int         tileStart = IMUL(blockIdx.y, COLUMN_TILE_H);
    const int           tileEnd = tileStart + COLUMN_TILE_H - 1;
    const int        apronStart = tileStart - KERNEL_RADIUS;
    const int          apronEnd = tileEnd   + KERNEL_RADIUS;

    //Clamp tile and apron limits by image borders
    const int    tileEndClamped = min(tileEnd, dataH - 1);
    const int apronStartClamped = max(apronStart, 0);
    const int   apronEndClamped = min(apronEnd, dataH - 1);

    //Current column index
    const int       columnStart = IMUL(blockIdx.x, COLUMN_TILE_W) + threadIdx.x;

    //Shared and global memory indices for current column
    int smemPos = IMUL(threadIdx.y, COLUMN_TILE_W) + threadIdx.x;
    int gmemPos = IMUL(apronStart + threadIdx.y, dataW) + columnStart;
    //Cycle through the entire data cache
    //Load global memory values, if indices are within the image borders,
    //or initialize with zero otherwise
    for(int y = apronStart + threadIdx.y; y <= apronEnd; y += blockDim.y){
        data[smemPos] =
        ((y >= apronStartClamped) && (y <= apronEndClamped)) ?
        d_Data[gmemPos] : 0;
        smemPos += smemStride;
        gmemPos += gmemStride;
    }

    //Ensure the completness of the loading stage
    //because results, emitted by each thread depend on the data,
    //loaded by another threads
    __syncthreads();
    //Shared and global memory indices for current column
    smemPos = IMUL(threadIdx.y + KERNEL_RADIUS, COLUMN_TILE_W) + threadIdx.x;
    gmemPos = IMUL(tileStart + threadIdx.y , dataW) + columnStart;
    //Cycle through the tile body, clamped by image borders
    //Calculate and output the results
    for(int y = tileStart + threadIdx.y; y <= tileEndClamped; y += blockDim.y){
        float sum = 0;
'''
originalLoop = '''
        for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
            sum += data[smemPos + IMUL(k, COLUMN_TILE_W)] *
            d_Kernel_columns[KERNEL_RADIUS - k];
'''
unrolledLoop = ''
for k in range(-KERNEL_RADIUS,  KERNEL_RADIUS+1):
    loopTemplate = string.Template('sum += data[smemPos + IMUL($k, COLUMN_TILE_W)] * d_Kernel_columns[KERNEL_RADIUS - $k];\n')
    unrolledLoop += loopTemplate.substitute(k=k)

#print unrolledLoop
template += unrolledLoop if UNROLL_INNER_LOOP else originalLoop
template += '''
        d_Result[gmemPos] = sum;
        //d_Result[gmemPos] = 128;
        smemPos += smemStride;
        gmemPos += gmemStride;
    }
}
'''
template = string.Template(template)
code = template.substitute(KERNEL_RADIUS = KERNEL_RADIUS,
                           KERNEL_W = KERNEL_W,
                           COLUMN_TILE_H=COLUMN_TILE_H,
                           COLUMN_TILE_W=COLUMN_TILE_W,
                           ROW_TILE_W=ROW_TILE_W,
                           KERNEL_RADIUS_ALIGNED=KERNEL_RADIUS_ALIGNED)

module = SourceModule(code)
convolutionRowGPU = module.get_function('convolutionRowGPU')
convolutionColumnGPU = module.get_function('convolutionColumnGPU')
d_Kernel_rows = module.get_global('d_Kernel_rows')[0]
d_Kernel_columns = module.get_global('d_Kernel_columns')[0]

# Helper functions for computing alignment...
def iDivUp(a, b):
    # Round a / b to nearest higher integer value
    a = numpy.int32(a)
    b = numpy.int32(b)
    return (a / b + 1) if (a % b != 0) else (a / b)

def iDivDown(a, b):
    # Round a / b to nearest lower integer value
    a = numpy.int32(a)
    b = numpy.int32(b)
    return a / b;

def iAlignUp(a, b):
    # Align a to nearest higher multiple of b
    a = numpy.int32(a)
    b = numpy.int32(b)
    return (a - a % b + b) if (a % b != 0) else a

def iAlignDown(a, b):
    # Align a to nearest lower multiple of b
    a = numpy.int32(a)
    b = numpy.int32(b)
    return a - a % b

def gaussian_kernel(width = KERNEL_W, sigma = 4.0):
    assert width == numpy.floor(width),  'argument width should be an integer!'
    radius = (width - 1)/2.0
    x = numpy.linspace(-radius,  radius,  width)
    x = numpy.float32(x)
    sigma = numpy.float32(sigma)
    filterx = x*x / (2 * sigma * sigma)
    filterx = numpy.exp(-1 * filterx)
    assert filterx.sum()>0,  'something very wrong if gaussian kernel sums to zero!'
    filterx /= filterx.sum()
    return filterx

def derivative_of_gaussian_kernel(width = KERNEL_W, sigma = 4):
    assert width == numpy.floor(width),  'argument width should be an integer!'
    radius = (width - 1)/2.0
    x = numpy.linspace(-radius,  radius,  width)
    x = numpy.float32(x)
    # The derivative of a gaussian is really just a gaussian times x, up to scale.
    filterx = gaussian_kernel(width,  sigma)
    filterx *= x
    # Rescale so that filter returns derivative of 1 when applied to x:
    scale = (x * filterx).sum()
    filterx /= scale
    # Careful with sign; this will be uses as a ~convolution kernel, so should start positive, then go negative.
    filterx *= -1.0
    return filterx

def test_derivative_of_gaussian_kernel():
    width = 20
    sigma = 10.0
    filterx = derivative_of_gaussian_kernel(width,  sigma)
    x = 2 * numpy.arange(0, width)
    x = numpy.float32(x)
    response = (filter * x).sum()
    assert abs(response - (-2.0)) < .0001, 'derivative of gaussian failed scale test!'
    width = 19
    sigma = 10.0
    filterx = derivative_of_gaussian_kernel(width,  sigma)
    x = 2 * numpy.arange(0, width)
    x = numpy.float32(x)
    response = (filterx * x).sum()
    assert abs(response - (-2.0)) < .0001, 'derivative of gaussian failed scale test!'

def convolution_cuda(sourceImage,  filterx,  filtery):
    # Perform separable convolution on sourceImage using CUDA.
    # Operates on floating point images with row-major storage.
    destImage = sourceImage.copy()
    assert sourceImage.dtype == 'float32',  'source image must be float32'
    (imageHeight,  imageWidth) = sourceImage.shape
    assert filterx.shape == filtery.shape == (KERNEL_W, ) ,  'Kernel is compiled for a different kernel size! Try changing KERNEL_W'
    filterx = numpy.float32(filterx)
    filtery = numpy.float32(filtery)
    DATA_W = iAlignUp(imageWidth, 16);
    DATA_H = imageHeight;
    BYTES_PER_WORD = 4;  # 4 for float32
    DATA_SIZE = DATA_W * DATA_H * BYTES_PER_WORD;
    KERNEL_SIZE = KERNEL_W * BYTES_PER_WORD;
    # Prepare device arrays
    destImage_gpu = cuda.mem_alloc_like(destImage)
    sourceImage_gpu = cuda.mem_alloc_like(sourceImage)
    intermediateImage_gpu = cuda.mem_alloc_like(sourceImage)
    cuda.memcpy_htod(sourceImage_gpu, sourceImage)
    cuda.memcpy_htod(d_Kernel_rows,  filterx) # The kernel goes into constant memory via a symbol defined in the kernel
    cuda.memcpy_htod(d_Kernel_columns,  filtery)
    # Call the kernels for convolution in each direction.
    blockGridRows = (iDivUp(DATA_W, ROW_TILE_W), DATA_H)
    blockGridColumns = (iDivUp(DATA_W, COLUMN_TILE_W), iDivUp(DATA_H, COLUMN_TILE_H))
    threadBlockRows = (KERNEL_RADIUS_ALIGNED + ROW_TILE_W + KERNEL_RADIUS, 1, 1)
    threadBlockColumns = (COLUMN_TILE_W, 8, 1)
    DATA_H = numpy.int32(DATA_H)
    DATA_W = numpy.int32(DATA_W)
    grid_rows = tuple([int(e) for e in blockGridRows])
    block_rows = tuple([int(e) for e in threadBlockRows])
    grid_cols = tuple([int(e) for e in blockGridColumns])
    block_cols = tuple([int(e) for e in threadBlockColumns])
    convolutionRowGPU(intermediateImage_gpu,  sourceImage_gpu,  DATA_W,  DATA_H,  grid=grid_rows,  block=block_rows)
    convolutionColumnGPU(destImage_gpu,  intermediateImage_gpu,  DATA_W,  DATA_H,  numpy.int32(COLUMN_TILE_W * threadBlockColumns[1]),  numpy.int32(DATA_W * threadBlockColumns[1]),  grid=grid_cols,  block=block_cols)

    # Pull the data back from the GPU.
    cuda.memcpy_dtoh(destImage,  destImage_gpu)
    return destImage

def test_convolution_cuda():
    # Test the convolution kernel.
    # Generate or load a test image
    original = numpy.random.rand(768,  1024) * 255
    original = numpy.float32(original)
    # You probably want to display the image using the tool of your choice here.
    filterx = gaussian_kernel()
    destImage = original.copy()
    destImage[:] = numpy.nan
    destImage = convolution_cuda(original,  filterx,  filterx)
    # You probably want to display the result image using the tool of your choice here.
    print('Done running the convolution kernel!')

if __name__ == '__main__':
    test_convolution_cuda()
    #test_derivative_of_gaussian_kernel()
    boo = input('Pausing so you can look at results... <Enter> to finish...')

