import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp
cimport AVX
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef r_pr2(float sigma_s,
                        float sigma_r,
                        np.float32_t [:, :] input,
                        int lx,
                        int ly,
                        np.float32_t [:, :] output,
                        int lw,
                        int num_threads):

	cdef:
		int i, j, k, l, _
		np.float32_t [:,:] local_input
		float sumg, I, coef, local_output
		int pan = 2 * lw + 1

    #loop on the pixels for output imagge
	for i in range(lx,2*lx):
		for j in range(ly,2*ly):
			local_input = input[i-lw:i+lw+1, j-lw:j+lw+1]
			sumg = 0.
			local_output = 0.
			I = local_input[lw,lw]
			#prange
            # Within the window for 1 pixel
            #loop over rows
			for k in prange(pan, nogil=True, schedule = 'static', chunksize =1, num_threads=num_threads):
				coef = <float> ((k-lw)*(k-lw))
				coef+= ((local_input[k,lw]-I)*(local_input[k,lw]-I))
				coef = coef * (-.5/sigma_s**2)
				coef = exp(coef)
				local_output += local_input[k,lw] * coef
				sumg += coef
				for l in range(1, lw+1, 8):
					for _ in range(l,l+8):
						local_output += local_input[k,lw+_] * exp(-.5/sigma_s**2*((local_input[k,lw+_]-I)*(local_input[k,lw+_]-I)+<float> ((k-lw)*(k-lw)+_*_)))
						sumg += exp(-.5/sigma_s**2*((local_input[k,lw+_]-I)*(local_input[k,lw+_]-I)+<float> ((k-lw)*(k-lw)+_*_)))
				for l in range(0, lw, 8):
					for _ in range(l,l+8):
						local_output += local_input[k,_] * exp(-.5/sigma_s**2*(<float> ((local_input[k,_]-I)*(local_input[k,_]-I)+(k-lw)*(k-lw)+(_-lw)*(_-lw))))
						sumg +=exp(-.5/sigma_s**2*(<float> ((local_input[k,_]-I)*(local_input[k,_]-I)+(k-lw)*(k-lw)+(_-lw)*(_-lw))))
            #update output
			output[i-lx,j-ly] = local_output / sumg
