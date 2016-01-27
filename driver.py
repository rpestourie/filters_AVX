import sys
import os.path
sys.path.append(os.path.join('.', 'util'))

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

# import mandelbrot
from timer import Timer

import pylab as plt
import numpy as np

from gaussian_filter import cython_gaussian, cython_bilateral
from gaussian_filter import cython_gaussian_pr, cython_bilateral_pr
from gaussian_filter import cython_gaussian_pr2
from gaussian_filter import cython_gaussian_pr3
from AVX_filtercode import cython_bilateral_AVX_pr

from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter

truncate = 4.
sigma = 4.
imagename = 'test.png'

# get ideal side sizes from arguments
lw = int(truncate*sigma+0.5)
pan = 2*lw+1

# for now we take values on the right until it fills a multiple of 8
# AVX is in x direction (in the matrix)
span = (pan // 8 + 1) * 8

# load image (only one color now)
f = misc.imread(imagename)
f = f[:,:,0]
output = f*0.

# add span-1 the -1 is because there is always at least one from the original image
lx, ly = f.shape
spx, spy = np.array([lx + span-1, ly + pan-1])

# add halo
input = np.zeros((3*lx,3*ly))
for i in range(3):
    for j in range(3):
            input[i*lx:(i+1)*lx, j*ly:(j+1)*ly] = f

output =  np.array(output, np.float32)
input = np.array(input, np.float32)

# start = time.time()
# cython_gaussian(input, output, lw, pan, lx, ly, sigma)
# print "cython gaussian", time.time() - start
#
# np.savetxt("foo.csv", output, delimiter=",")
#
# plt.imshow(output)
# plt.show()
#
# output = f*0.
# output2 =  np.array(output, np.float32)
#
# start = time.time()
# cython_bilateral(input, output2, lw, pan, lx, ly, sigma)
# print "cython bilateral", time.time() - start
#
# np.savetxt("foo2.csv", output2, delimiter=",")
#
# plt.imshow(output2)
# plt.show()

# output = f*0.
# output3 =  np.array(output, np.float32)
#
# start = time.time()
# cython_gaussian_pr(input, output3, lw, pan, lx, ly, sigma,1)
# print "cython bilateral prange", time.time() - start
#
# np.savetxt("foo3.csv", output3, delimiter=",")
#
# plt.imshow(output3)
# plt.show()
#
# output = f*0.
# output4 =  np.array(output, np.float32)
#
# start = time.time()
# cython_gaussian_pr2(input, output4, lw, pan, lx, ly, sigma)
# print "cython bilateral prange", time.time() - start
#
# np.savetxt("foo4.csv", output4, delimiter=",")
#
# plt.imshow(output4[:10,:10])
# plt.show()
#
# output = f*0.
# output5 =  np.array(output, np.float32)
#
# start = time.time()
# cython_gaussian_pr3(input, output5, lw, pan, lx, ly, sigma,2)
# print "cython bilateral prange", time.time() - start
#
# np.savetxt("foo5.csv", output5, delimiter=",")
#
# plt.imshow(output5)
# plt.show()
#
# output = f*0.
# output5 =  np.array(output, np.float32)
#
# start = time.time()
# cython_gaussian_pr3(input, output5, lw, pan, lx, ly, sigma,1)
# print "cython bilateral prange", time.time() - start
#
# np.savetxt("foo5.csv", output5, delimiter=",")
#
# plt.imshow(output5)
# plt.show()

# from AVX_filtercode import cython_gaussian_AVX, cython_gaussian_AVX_pr, cython_bilateral_AVX_pr
# output = f*0.
# output5 =  np.array(output, np.float32)
# start = time.time()
# cython_gaussian_AVX(input, output5, lw, pan, lx, ly, sigma,1)
# print "AVX gaussian gil", time.time() - start
# plt.imshow(output5)
# plt.show()
#
# output = f*0.
# output5 =  np.array(output, np.float32)
# start = time.time()
# cython_gaussian_AVX_pr(input, output5, lw, pan, lx, ly, sigma,1)
# print "AVX gaussian th 1", time.time() - start
# plt.imshow(output5)
# plt.show()
#
# output = f*0.
# output5 =  np.array(output, np.float32)
# start = time.time()
# cython_gaussian_AVX_pr(input, output5, lw, pan, lx, ly, sigma,2)
# print "AVX gaussian th 2", time.time() - start
# plt.imshow(output5)
# plt.show()

output = f*0.
output5 =  np.array(output, np.float32)
start = time.time()
cython_bilateral_AVX_pr(input, output5, lw, pan, lx, ly, sigma,2)
print "AVX bilateral th 2", time.time() - start
plt.imshow(output5)
# plt.show()
print output5.shape
print output.shape
print 'differences', np.linalg.norm(output5 - input[0:lx, 0:ly])
