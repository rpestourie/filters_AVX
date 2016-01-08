from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
from Gaussianfilter2D import Gaussianfilter2D


# --------------------------------------------
# initialization - define image
# --------------------------------------------
imagename = 'small.png'

# load image (only one color now)
f = misc.imread(imagename)
f = f[:,:,0]



# --------------------------------------------
# Gaussian filter
# --------------------------------------------

# create a instance gaussian filter
gb = Gaussianfilter2D(sigma = 4.0, truncate = 4.0, mode = 'constant', cval = 200.0)


# --------------------------------------------
# print the variables
# --------------------------------------------


print 'lw', gb.lw 
print 'pan', gb.pan
print 'span', gb.span
print 'f.shape', f.shape

gb.filter_python( f)

print 'error compared to scipy', gb.error_

print 'run time scipy' , gb.run_time_benchmark_
print 'run time python' , gb.run_time_