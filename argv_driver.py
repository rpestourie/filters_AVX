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

#import another_bilateral_implementation
from another_bilateral_implementation import bilateral_filter

from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter

#setup parameters
im_size = 128
sigma_r = 2.
sigma_s = 2.
truncate = 4.
lw = int(max(sigma_r, sigma_s)*truncate)

# picture
picture = np.random.random((im_size, im_size))

#preprocess image
imsize = np.shape(picture)
input_im = np.zeros((lw + imsize[0] + lw, lw + imsize[1] + lw + 8 - (2 * lw + 1) % 8))
input_im[:(lw + imsize[0] + lw), :(lw + imsize[1] + lw)] = np.pad(picture, ((lw, lw), (lw, lw)), mode = 'reflect')
plt.imshow(input_im);

#run filter
imsize0 = imsize[0]
imsize1 = imsize[1]
output = picture*0.
output5 =  np.array(output, np.float32)
input_im5 = np.array(input_im, np.float32)
start = time.time()
bilateral_filter(sigma_s,
                        sigma_r,
                        input_im5,
                        imsize0,
			imsize1,
                        output5,
                        lw)
print "bilateral", time.time() - start
plt.imshow(output5)
plt.show()
