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
