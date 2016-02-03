#!/usr/bin/python
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


from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter


# for truncate in range(10):
    # print("truncate= ", truncate)

#setup parameters
im_size = 512
sigma_r = 2.
sigma_s = 2.
truncate = 4.
lw = int(max(sigma_r, sigma_s)*truncate)

# picture
# picture = np.identity(512)
picture = np.random.random((im_size, im_size))

#preprocess image
imsize = np.shape(picture)
input_im = np.zeros((lw + imsize[0] + lw, lw + imsize[1] + lw + 8 - (2 * lw + 1) % 8))
input_im[:(lw + imsize[0] + lw), :(lw + imsize[1] + lw)] = np.pad(picture, ((lw, lw), (lw, lw)), mode = 'reflect')
plt.imshow(input_im);

# from another_bilateral_implementation import bilateral_filter
# #run filter
# imsize0 = imsize[0]
# imsize1 = imsize[1]
# output = picture*0.
# output5 =  np.array(output, np.float32)
# input_im5 = np.array(input_im, np.float32)
# start = time.time()
# bilateral_filter(sigma_s,
#                         sigma_r,
#                         input_im5,
#                         imsize0,
# 			imsize1,
#                         output5,
#                         lw)
# print "bilateral", time.time() - start
#
# from another_bilateral_implementation_pr import bilateral_filter_pr
# #run filter
# imsize0 = imsize[0]
# imsize1 = imsize[1]
# output = picture*0.
# output5 =  np.array(output, np.float32)
# input_im5 = np.array(input_im, np.float32)
# start = time.time()
# bilateral_filter_pr(sigma_s,
#                         sigma_r,
#                         input_im5,
#                         imsize0,
# 			imsize1,
#                         output5,
#                         lw,
#                         1)
# print "bilateral_pr 1 th", time.time() - start
# # plt.imshow(output5)
# # plt.show()
#
# #run filter
# imsize0 = imsize[0]
# imsize1 = imsize[1]
# output = picture*0.
# output5 =  np.array(output, np.float32)
# input_im5 = np.array(input_im, np.float32)
# start = time.time()
# bilateral_filter_pr(sigma_s,
#                         sigma_r,
#                         input_im5,
#                         imsize0,
# 			imsize1,
#                         output5,
#                         lw,
#                         2)
# print "bilateral_pr 2 th", time.time() - start
# # plt.imshow(output5)
# # plt.show()
#
# from tst import bilateral_filter_pr_j
# #run filter
# imsize0 = imsize[0]
# imsize1 = imsize[1]
# output = picture*0.
# output5 =  np.array(output, np.float32)
# input_im5 = np.array(input_im, np.float32)
# start = time.time()
# bilateral_filter_pr_j(sigma_s,
#                         sigma_r,
#                         input_im5,
#                         imsize0,
# 			imsize1,
#                         output5,
#                         lw)
# print "bilateral_pr_j", time.time() - start
# # plt.imshow(output5)
# # plt.show()
#
# from tst import bilateral_filter_pr_i
# #run filter
# imsize0 = imsize[0]
# imsize1 = imsize[1]
# output = picture*0.
# output5 =  np.array(output, np.float32)
# input_im5 = np.array(input_im, np.float32)
# start = time.time()
# bilateral_filter_pr_j(sigma_s,
#                         sigma_r,
#                         input_im5,
#                         imsize0,
# 			imsize1,
#                         output5,
#                         lw)
# print "bilateral_pr_i", time.time() - start
# # plt.imshow(output5)
# # plt.show()
#
#
# from another_bilateral_implementation_AVX import bilateral_filter_AVX
# #run filter
# imsize0 = imsize[0]
# imsize1 = imsize[1]
# output = picture*0.
# output5 =  np.array(output, np.float32)
# input_im5 = np.array(input_im, np.float32)
# start = time.time()
# bilateral_filter_AVX(sigma_s,
#                         sigma_r,
#                         input_im5,
#                         imsize0,
# 			imsize1,
#                         output5,
#                         lw)
# print "bilateral_AVX", time.time() - start
# # plt.imshow(output5)
# # plt.show()

from offset import bilateral_filter_pr_j_offset
#run filter

offsets = [0]
for k in range(5,6):
    offsets.append(2**5-1)
for offset in offsets:
    imsize0 = imsize[0]
    imsize1 = imsize[1]
    output = picture*0.
    output5 =  np.array(output, np.float32)
    input_im5 = np.array(input_im, np.float32)
    start = time.time()
    bilateral_filter_pr_j_offset(sigma_s,
                            sigma_r,
                            input_im5,
                            imsize0,
    			imsize1,
                            output5,
                            lw,
                            offset)
    print "bilateral_AVX", time.time() - start
    if offset == 0:
        plt.subplot(121)
        plt.imshow(output5)
plt.subplot(122)
plt.imshow(output5)
plt.show()
