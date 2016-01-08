from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter


# --------------------------------------------
# initialization - define image
# --------------------------------------------


# function input
truncate = 4.
sigma = 4.
imagename = 'small.png'

# get ideal side sizes from arguments
lw = int(truncate*sigma+0.5)
pan = 2*lw+1

# for now we take values on the right until it fills a multiple of 8
# AVX is in x direction (in the matrix)
span = (pan // 8 + 1) * 8
# span = 40

# load image (only one color now)
f = misc.imread(imagename)
f = f[:,:,0]

# create the zero matrix for output
output = f*0.0

# add span-1 the -1 is because there is always at least one from the original image
lx, ly = f.shape
spx, spy = np.array([lx + span-1, ly + pan-1])

# --------------------------------------------
# print the variables
# --------------------------------------------


print 'lw', lw 
print 'pan', pan
print 'span', span
print 'f.shape', f.shape
print 'lx, ly', lx, ly
print 'spx, spy', spx, spy

# --------------------------------------------
im_test = np.zeros((3*lx,3*ly))
for i in range(3):	
    for j in range(3):
            im_test[i*lx:(i+1)*lx, j*ly:(j+1)*ly] = f


# --------------------------------------------
# Gaussian filter
# --------------------------------------------
print np.linalg.norm([1,1])




