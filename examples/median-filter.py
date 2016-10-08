import numpy as np
from scipy.misc import imread, imsave
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

# Maximum thread size for GPU is dependent on GPU, but normally 512.
# Threads per block should be a multiple of 32.
# Block and Grid Size is dependent on the image.
# This example uses a 256x256 pixel image. A 2D block (16x16) and a 1D grid (256,1) is used

#Read in image
img = imread('noisyImage.jpg', flatten=True).astype(np.float32)


mod = SourceModule('''
__host__ __device__ void sort(int *a, int *b, int *c) {
    int swap;
    if(*a > *b) {
        swap = *a;
        *a = *b;
        *b = swap;
    }
    if(*a > *c) {
        swap = *a;
        *a = *c;
        *c = swap;
    }
    if(*b > *c) {
        swap = *b;
        *b = *c;
        *c = swap;
    }
}
__global__ void medianFilter(float *result, float *img, int w, int h) {
    //2D Blocks, 1D Grid. Finding respective index
    int i = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    // Keeping the edge pixels the same
    if (i < w || i > w * (h-1)-1 || i % (w-1) == 0 ) {
        result[i] = img[i];
    }
    else {
        int pixel00, pixel01, pixel02, pixel10, pixel11, pixel12, pixel20, pixel21, pixel22;
        pixel00 = img[i - 1 - w];
        pixel01 = img[i- w];
        pixel02 = img[i + 1 - w];
        pixel10 = img[i - 1];
        pixel11 = img[i];
        pixel12 = img[i + 1];
        pixel20 = img[i - 1 + w];
        pixel21 = img[i + w];
        pixel22 = img[i + 1 + w];
        //sort the rows
        sort( &(pixel00), &(pixel01), &(pixel02) );
        sort( &(pixel10), &(pixel11), &(pixel12) );
        sort( &(pixel20), &(pixel21), &(pixel22) );
        //sort the columns
        sort( &(pixel00), &(pixel10), &(pixel20) );
        sort( &(pixel01), &(pixel11), &(pixel21) );
        sort( &(pixel02), &(pixel12), &(pixel22) );
        //sort the diagonal
        sort( &(pixel00), &(pixel11), &(pixel22) );
        // median is the the middle value of the diagonal
        result[i] = pixel11;
    }
}''')