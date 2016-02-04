import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp
cimport AVX
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)

cpdef r_r2(float sigma_s,
                        float sigma_r,
                        np.float32_t [:, :] input,
                        int lx,
                        int ly,
                        np.float32_t [:, :] output,
                        int lw):


	cdef:
		int i, j, k, l, _
		np.float32_t [:,:] local_input
		float sumg, coef, I, local_value
		int pan = 2 * lw + 1

    #loop on the pixels for output imagge
	for i in range(lx,2*lx):
		for j in range(ly,2*ly):
			local_input = input[i-lw:i+lw+1, j-lw:j+lw+1] #to avoid fetching in the array
			sumg = 0 #normalization term
			I = local_input[lw,lw]
            # Within the window for 1 pixel
            #loop over rows
			for k in range(pan):
                # loop in the line in 3 part: the center
                # due to properties of the exp function we just sum the contributtions within the exp
				coef = <float> ((k-lw)*(k-lw))
				coef+= ((local_input[k,lw]-I)*(local_input[k,lw]-I))
				coef = coef * (-.5/sigma_s**2)
				coef = exp(coef)
				output[i-lx,j-ly] += local_input[k,lw] * coef
				sumg += coef
                #the right part of the line (suits avx)
				for l in range(1, lw+1, 8):
					for _ in range(l,l+8):
						local_value = local_input[k,lw+_]
						coef = exp(-.5/sigma_s**2*((local_value-I)*(local_value-I)+<float> ((k-lw)*(k-lw)+_*_)))
						output[i-lx,j-ly] += local_value * coef
						sumg += coef
                #the left part of the line (suits avx)
				for l in range(0, lw, 8):
					for _ in range(l,l+8):
						local_value = local_input[k,_]
						coef = exp(-.5/sigma_s**2*((local_value-I)*(local_value-I)+<float> ((k-lw)*(k-lw)+(_-lw)*(_-lw))))
						output[i-lx,j-ly] += local_value * coef
						sumg += coef
            #update output
			output[i-lx,j-ly] /= sumg
