#!/usr/bin/python
import sys
import os.path
sys.path.append(os.path.join('.', 'util'))

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

from timer import Timer

import pylab as plt
import numpy as np


#setup parameters
im_size = int(sys.argv[1])
sigma_r = int(sys.argv[2])
sigma_s = int(sys.argv[3])
truncate = int(sys.argv[4])
lw = int(max(sigma_r, sigma_s)*truncate)

picture = np.random.random((im_size, im_size))

#preprocess image
imsize = np.shape(picture)
input_im = np.zeros((lw + imsize[0] + lw, lw + imsize[1] + lw + 8 - (2 * lw + 1) % 8))
input_im[:(lw + imsize[0] + lw), :(lw + imsize[1] + lw)] = np.pad(picture, ((lw, lw), (lw, lw)), mode = 'reflect')
plt.imshow(input_im);

from timefunction import time_update

from r_pr2 import r_pr2

fn = r_pr2

time_update(fn, truncate, imsize, picture, input_im, sigma_r, sigma_s, lw, 2)
