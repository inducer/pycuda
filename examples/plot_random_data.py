# simple module to show the plotting of random data

from matplotlib.pylab import *
import pycuda.curandom as curandom

size = 1000

#random data generated on gpu
a = curandom.rand((size,)).get()

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
