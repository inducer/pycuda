#simple module to show the ploting of random data

import pycuda.simplearray as cuda
from matplotlib.pylab import *

size = 1000

#random data generated on gpu
a = cuda.array(size).fill_random()


subplot(211)
plot(a)
grid(True)
ylabel('plot - gpu')

subplot(212)
hist(a, 100)
grid(True)
ylabel('histogram - gpu')

#and save it
savefig('plot-random-data')