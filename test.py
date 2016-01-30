# ------------------------------------------------------------
# import modules to compile cython and AVX
# ------------------------------------------------------------

import sys
import os.path
sys.path.append(os.path.join('.', 'util'))

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()


from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
from Gaussianfilter2D_class import Gaussianfilter2D

30
# --------------------------------------------
# initialization - define image
# --------------------------------------------
imagename = 'test.png'

# load image (only one color now)
f = misc.imread(imagename)
if imagename == 'small.png':
	f = f[:,:,0]
else:
	pass



# --------------------------------------------
# Gaussian filter
# --------------------------------------------

# create a instance gaussian filter
gb = Gaussianfilter2D(sigma = 8.0, truncate = 8.0, mode = 'reflect', cval = 0.0)


# --------------------------------------------
# print the variables
# --------------------------------------------


print 'lw', gb.lw 
print 'pan', gb.pan
print 'span', gb.span
print 'f.shape', f.shape

gb.filter_cython(f)

print 'error compared to scipy', gb.error_

print 'run time scipy' , gb.run_time_benchmark_
print 'run time python' , gb.run_time_

print 'error other compared to scipy', gb.error_

print 'kernel.shape', gb.kernel_.shape




# # --------------------------------------------
# # plot the images
# # --------------------------------------------

fig, ax = plt.subplots(1,2)
ax[0].imshow(gb.image_);
ax[0].set_title('Python algorithm');
ax[1].imshow(gb.image_benchmark_);
ax[1].set_title('Scipy library');
plt.show()


