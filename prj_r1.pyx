import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp
cimport AVX
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)

cdef float gaussian(float x2,
                float sigma) nogil:
    """get gaussian coefficient with x^2 as input"""
    return exp(-.5*x2/sigma**2)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)

cpdef prj_r1(float sigma_s,
                        float sigma_r,
                        np.float32_t [:, :] input_im,
                        int imsize0,
                        int imsize1,
                        np.float32_t [:, :] output,
                        int lw,
                        int num_thread):

    cdef:
        int i,j,k,l,cur, ii,ixx, xx, iyy, yy,_
        np.float32_t [:, :] window
        np.float32_t [:] local_input
        int windowsize0, windowsize1
        float sumg, local_output, I_pixel, coef
        np.float32_t [:] partial_output, partial_sumg

    local_input = np.zeros(8, np.float32)
    window = np.zeros((2 * lw +1, lw + 1 + lw + 8 - (2 * lw + 1) % 8), np.float32)
    windowsize0, windowsize1 = np.shape(window)


    # to implement with multi process
    for i in range(lw, imsize0 + lw):
        #initalize parameters
        partial_output = np.zeros(imsize1, np.float32)
        partial_sumg = np.zeros(imsize1, np.float32)

        for j in prange(lw, imsize1 + lw, nogil = True, num_threads= num_thread):

            # window contining halo
            for iyy, yy in enumerate(range(j - lw, j + 1 + lw + 8 - (2 * lw + 1) % 8)):
                for ixx, xx in enumerate(range(i - lw, i + lw + 1)):
                    window[ixx, iyy] = input_im[xx, yy]
#             window = input_im[i - lw : i + lw + 1, j - lw : j + 1 + lw + 8 - (2 * lw + 1) % 8]
#             assert np.shape(window) == (2 * lw +1, lw + 1 + lw + 8 - (2 * lw + 1) % 8)


            # initialize normalization term and output term
            I_pixel = window[lw, lw]

            for k in range(windowsize0):
                for l in range(0, windowsize1, 8):
                    # take local_input which will become an AVX 8-array
                    for ii in range(8):
                        local_input[ii] = window[k,l+ii]
#                     assert np.size(local_input) == 8

                    for cur in range(8):
                        # don't consider the pixel if it is outside the window
                        if l + cur < 2 * lw + 1:
                            coef = gaussian((k - lw)**2+(l + cur - lw)**2, sigma_s)
                            coef *= gaussian((<float> local_input[cur] - I_pixel)**2, sigma_r)
                            partial_output[j-lw] += coef * <float> local_input[cur]
                            partial_sumg[j-lw] += coef

        for _ in range(imsize1):
            output[i-lw, _] = partial_output[_] / partial_sumg[_]
