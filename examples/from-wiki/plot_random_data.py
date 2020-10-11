#!python 
# simple module to show the plotting of random data

import pycuda.autoinit
import pycuda.curandom as curandom

size = 1000
a = curandom.rand((size,)).get()

from matplotlib.pylab import *
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


