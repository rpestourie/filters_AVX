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
cpdef r_r1(float sigma_s,
                        float sigma_r,
                        np.float32_t [:, :] input_im,
                        int imsize0,
                        int imsize1,
                        np.float32_t [:, :] output,
                        int lw):

    cdef:
        int i,j,k,l,cur
        np.float32_t [:, :] window
        np.float32_t [:] local_input
        int windowsize0, windowsize1
        float sumg, local_output, I_pixel, coef

    # to implement with multi process
    for i in range(lw, imsize0 + lw):
        for j in range(lw, imsize1 + lw):

            # window contining halo
            window = input_im[i - lw : i + lw + 1, j - lw : j + 1 + lw + 8 - (2 * lw + 1) % 8]
            if np.shape(window) != (2 * lw +1, lw + 1 + lw + 8 - (2 * lw + 1) % 8):
                print np.shape(window) , (2 * lw +1, lw + 1 + lw + 8 - (2 * lw + 1) % 8)
            assert np.shape(window) == (2 * lw +1, lw + 1 + lw + 8 - (2 * lw + 1) % 8)
            windowsize0, windowsize1 = np.shape(window)

            # initialize normalization term and output term
            sumg = 0
            local_output = 0
            I_pixel = window[lw, lw]

            for k in range(windowsize0):
                for l in range(0, windowsize1, 8):
                    # take local_input which will become an AVX 8-array
                    local_input = window[k,l:l+8]
                    assert np.size(local_input) == 8

                    for cur in range(8):
                        # don't consider the pixel if it is outside the window
                        if l + cur < 2 * lw + 1:
                            coef = gaussian((k - lw)**2+(l + cur - lw)**2, sigma_s)
                            coef *= gaussian((<float> local_input[cur] - I_pixel)**2, sigma_r)
                            local_output += coef * <float> local_input[cur]
                            sumg += coef

            output[i-lw, j-lw] = local_output / sumg
