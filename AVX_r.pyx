import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp
cimport AVX
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)

cpdef AVX_r(float sigma_s,
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
        np.float32_t [:] partial_output, partial_sumg
        np.float32_t [:] output_array, coef_array
        AVX.float8 AVX_I, AVX_local_input, AVX_coef, AVX_coef1, AVX_coef2, AVX_output, AVX_partial_output, AVX_sumg

    local_input = np.zeros(8, np.float32)
    window = np.zeros((2 * lw +1, 7 + lw + 1 + lw + 8 - (2 * lw + 1) % 8), np.float32)
    windowsize0, windowsize1 = np.shape(window)
    windowsize1 = windowsize1 - 7
    AVX_bil_coef = AVX.float_to_float8(-.5/sigma_r**2)
    output_array = np.zeros(8, dtype= np.float32)
    coef_array = np.zeros(8, dtype= np.float32)


    # to implement with multi process
    for i in range(lw, imsize0 + lw):
        #initalize parameters
        partial_output = np.zeros(imsize1, np.float32)
        partial_sumg = np.zeros(imsize1, np.float32)

        for j in prange(lw, imsize1 + lw, 7+1, nogil = True, num_threads=num_thread):

            # window contining halo (load for #7 pixels to take advantages of overlaps)
            for iyy, yy in enumerate(range(j - lw, 7 + j + 1 + lw + 8 - (2 * lw + 1) % 8)):
                for ixx, xx in enumerate(range(i - lw, i + lw + 1)):
                    window[ixx, iyy] = (
                    input_im[xx, yy])
#             window = input_im[i - lw : i + lw + 1, j - lw : j + 1 + lw + 8 - (2 * lw + 1) % 8]
#             assert np.shape(window) == (2 * lw +1, lw + 1 + lw + 8 - (2 * lw + 1) % 8)


            # initialize normalization term and output term
            # AVX_I = AVX.float_to_float8(0.)
            AVX_I = AVX.make_float8(window[lw, lw], window[lw, 1 + lw], window[lw, 2 + lw], window[lw, 3 + lw],
                                    window[lw, 4 + lw], window[lw, 5 + lw], window[lw, 6 + lw], window[lw, 7 + lw])
            AVX_output = AVX.float_to_float8(0.)
            AVX_sumg = AVX.float_to_float8(0.)

            for k in range(windowsize0):
                for l in range(windowsize1):
                    # take local_input which will become an AVX 8-array
                    AVX_local_input = AVX.make_float8(window[k, l], window[k,1 + l], window[k,2 + l], window[k,3 + l],
                                                    window[k,4 + l], window[k,5 + l], window[k,6 + l], window[k,7 + l])
                    #coef for bilateral
                    AVX_coef1 = AVX.sub(AVX_local_input, AVX_I)
                    AVX_coef1 = AVX.mul(AVX_coef1, AVX_coef1)
                    AVX_coef1 = AVX.mul(AVX_coef1, AVX_bil_coef)
                    #coef for gaussian
                    AVX_coef2 = AVX.float_to_float8(-.5/sigma_s**2*((k - lw)**2+(l - lw)**2))
                    #final coef
                    AVX_coef = AVX.add(AVX_coef1, AVX_coef2)
                    #compute the contribution from the pixels dones in parallel
                    AVX_partial_output = AVX.mul(AVX_local_input, AVX_coef)
                    #update sumg and output
                    AVX_output = AVX.add(AVX_output, AVX_partial_output)
                    AVX_sumg = AVX.add(AVX_sumg, AVX_coef)

                    for _ in range(8):
                        output_array[_] = <np.float32_t> (<np.float32_t *> &AVX_output)[_]
                        partial_output[_+j-lw] = output_array[_]
                        coef_array[_] = <np.float32_t> (<np.float32_t *> &AVX_sumg)[_]
                        partial_sumg[_+j-lw] = coef_array[_]
                        # partial_output[_+j-lw] = <np.float32_t> (<np.float32_t *> &AVX_ouput)[_]
                        # partial_sumg[_+j-lw] = <np.float32_t> (<np.float32_t *> &AVX_sumg)[_]

        for _ in range(imsize1):
            output[i-lw, _] = partial_output[_] / partial_sumg[_]
