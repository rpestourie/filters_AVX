import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp
cimport AVX
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)

cpdef prj_AVX(float sigma_s,
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
        np.float32_t [:] output_array, coef_array
        AVX.float8 AVX_local_input, AVX_x, AVX_y, AVX_gauss_coef, AVX_bil_coef, AVX_coef1, AVX_coef2, AVX_coef, AVX_ouput, AVX_I, AVX_MAXx, mask

    # local_input = np.zeros(8, np.float32)
    AVX_gauss_coef = AVX.float_to_float8(-.5/sigma_s**2) #coef gaussian
    AVX_bil_coef = AVX.float_to_float8(-.5/sigma_r**2) #coef bilateral
    AVX_MAXx = AVX.float_to_float8(<float> 2 * lw + 1)

    local_input = np.zeros(8, np.float32)
    window = np.zeros((2 * lw +1, lw + 1 + lw + 8 - (2 * lw + 1) % 8), np.float32)
    windowsize0, windowsize1 = np.shape(window)

    output_array = np.zeros(8, dtype= np.float32)
    coef_array = np.zeros(8, dtype= np.float32)

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
            sumg = 0
            local_output = 0
            for _ in range(8):
                output_array[_] = 0
                coef_array[_] = 0
            I_pixel = window[lw, lw]
            AVX_I =  AVX.float_to_float8(<float> I_pixel)

            for k in range(windowsize0):

                AVX_y =  AVX.float_to_float8(<float> (k-lw))

                for l in range(0, windowsize1, 8):
                    # take local_input which will become an AVX 8-array, with a rollover (using mod)
                    AVX_local_input = AVX.make_float8(window[k,(l)%(2 * lw + 1)], window[k,(l+1)%(2 * lw + 1)], window[k,(l+2)%(2 * lw + 1)], window[k,(l+3)%(2 * lw + 1)],
                    window[k,(l+4)%(2 * lw + 1)], window[k,(l+5)%(2 * lw + 1)],window[k,(l+6)%(2 * lw + 1)], window[k,(l+7)%(2 * lw + 1)])
                    AVX_x = AVX.make_float8(<float> l, <float> l+1, <float> l+2, <float> l+3, <float> l+4, <float> l+5, <float> l+6, <float> l+7)

                    #create mask
                    mask = AVX.less_than(AVX_x, AVX_MAXx)

                    # compute coefficient
                    #contribution from bilateral
                    AVX_coef1 = AVX.sub(AVX_local_input, AVX_I)
                    AVX_coef1 = AVX.mul(AVX_coef1, AVX_coef1)
                    AVX_coef1 = AVX.mul(AVX_coef1, AVX_bil_coef)
                    #contribution from gaussian filter
                    AVX_coef2 = AVX.sub(AVX_x, AVX.float_to_float8(<float> lw))
                    AVX_coef2 = AVX.mul(AVX_coef2, AVX_coef2)
                    AVX_coef2 = AVX.fmadd(AVX_y, AVX_y, AVX_coef2)
                    AVX_coef2 = AVX.mul(AVX_coef2, AVX_gauss_coef)
                    #final coef
                    AVX_coef = AVX.add(AVX_coef1, AVX_coef2)
                    # AVX_coef = AVX.exp(AVX_coef)

                    # output
                    AVX_ouput = AVX.mul(AVX_local_input, AVX.bitwise_and(mask, AVX_coef))

                    for _ in range(8):
                        output_array[_] = <np.float32_t> (<np.float32_t *> &AVX_ouput)[_]
                        partial_output[j-lw]  += output_array[_]
                        coef_array[_] = <np.float32_t> (<np.float32_t *> &AVX_coef)[_]
                        partial_sumg[j-lw] += coef_array[_]


        for _ in range(imsize1):
            output[i-lw, _] = partial_output[_] / partial_sumg[_]
