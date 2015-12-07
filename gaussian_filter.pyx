import numpy as np
cimport numpy as np
cimport cython
import numpy
from libc.math cimport exp
# cimport AVX
from cython.parallel import prange
from openmp cimport omp_lock_t, \
    omp_init_lock, omp_destroy_lock, \
    omp_set_lock, omp_unset_lock, omp_get_thread_num
from libc.stdlib cimport malloc, free

# lock helper functions
cdef void acquire(omp_lock_t *l) nogil:
	omp_set_lock(l)

cdef void release(omp_lock_t *l) nogil:
	omp_unset_lock(l)

# helper function to fetch and initialize several locks
cdef omp_lock_t *get_N_locks(int N) nogil:
	cdef:
		omp_lock_t *locks = <omp_lock_t *> malloc(N * sizeof(omp_lock_t))
		int idx

	if not locks:
		with gil:
			raise MemoryError()
	for idx in range(N):
			omp_init_lock(&(locks[idx]))

	return locks

cdef void free_N_locks(int N, omp_lock_t *locks) nogil:
	cdef int idx

	for idx in range(N):
		omp_destroy_lock(&(locks[idx]))

	free(<void *> locks)

@cython.boundscheck(False)
@cython.wraparound(False)



cpdef cython_bilateral(np.float32_t [:, :] input,
						np.float32_t [:, :] output,
						int lw,
						int pan,
						int lx,
						int ly,
						float sigma):


	cdef:
		int i, j, k, l, _
		np.float32_t [:,:] local_input
		float sumg, coef, I, local_value

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
				coef = coef * (-.5/sigma)
				coef = exp(coef)
				output[i-lx,j-ly] += local_input[k,lw] * coef
				sumg += coef
                #the right part of the line (suits avx)
				for l in range(1, lw+1, 8):
					for _ in range(l,l+8):
						local_value = local_input[k,lw+_]
						coef = exp(-.5/sigma*((local_value-I)*(local_value-I)+<float> ((k-lw)*(k-lw)+_*_)))
						output[i-lx,j-ly] += local_value * coef
						sumg += coef
                #the left part of the line (suits avx)
				for l in range(0, lw, 8):
					for _ in range(l,l+8):
						local_value = local_input[k,_]
						coef = exp(-.5/sigma*((local_value-I)*(local_value-I)+<float> ((k-lw)*(k-lw)+(_-lw)*(_-lw))))
						output[i-lx,j-ly] += local_value * coef
						sumg += coef
            #update output
			output[i-lx,j-ly] /= sumg

cpdef cython_gaussian(np.float32_t [:, :] input,
						np.float32_t [:, :] output,
						int lw,
						int pan,
						int lx,
						int ly,
						float sigma):

	cdef:
		int i, j, k, l, _
		np.float32_t [:,:] local_input
		float sumg, coef, local_value

    #loop on the pixels for output imagge
	for i in range(lx,2*lx):
		for j in range(ly,2*ly):
			local_input = input[i-lw:i+lw+1, j-lw:j+lw+1]
			sumg = 0
            # Within the window for 1 pixel
            #loop over rows
			for k in range(pan):
				coef = <float> ((k-lw)*(k-lw))
				coef = coef * (-.5/sigma)
				coef = exp(coef)
				output[i-lx,j-ly] += local_input[k,lw] * coef
				sumg += coef
				for l in range(1, lw+1, 8):
					for _ in range(l,l+8):
						local_value = local_input[k,lw+_]
						coef = exp(-.5/sigma*(<float> ((k-lw)*(k-lw)+_*_)))
						output[i-lx,j-ly] += local_value * coef
						sumg += coef
				for l in range(0, lw, 8):
					for _ in range(l,l+8):
						local_value = local_input[k,_]
						coef = exp(-.5/sigma*(<float> ((k-lw)*(k-lw)+(_-lw)*(_-lw))))
						output[i-lx,j-ly] += local_value * coef
						sumg += coef
            #update output
			output[i-lx,j-ly] /= sumg

cpdef cython_gaussian_pr(np.float32_t [:, :] input,
						np.float32_t [:, :] output,
						int lw,
						int pan,
						int lx,
						int ly,
						float sigma,
						int num_threads):

	cdef:
		int i, j, k, l, _
		np.float32_t [:,:] local_input
		float sumg, coef, I, local_value, coef1

#loop on the pixels for output imagge
	for i in range(lx,2*lx):
		for j in range(ly,2*ly):
			local_input = input[i-lw:i+lw+1, j-lw:j+lw+1]
			sumg = 0
			I = local_input[lw,lw]
            # Within the window for 1 pixel
            #loop over rows
			for k in range(pan):
				coef1 = <float> ((k-lw)*(k-lw))
				# coef+= ((local_input[k,lw]-I)*(local_input[k,lw]-I))
				coef1 = coef1 * (-.5/sigma)
				coef1 = exp(coef1)
				output[i-lx,j-ly] += local_input[k,lw] * coef1
				sumg += coef1
				for l in range(1, lw+1, 8):
                    #prange for right side of the line
					for _ in prange(l,l+8, nogil=True, schedule = 'static', chunksize =1, num_threads=num_threads):
						local_value = local_input[k,lw+_]
						coef = exp(-.5/sigma*(<float> ((k-lw)*(k-lw)+_*_)))
						output[i-lx,j-ly] += local_value * coef
						sumg += coef
				for l in range(0, lw, 8):
                    #prange for left side of the line
					for _ in prange(l,l+8, nogil=True, schedule = 'static', chunksize =1, num_threads=num_threads):
						local_value = local_input[k,_]
						coef = exp(-.5/sigma*(<float> ((k-lw)*(k-lw)+(_-lw)*(_-lw))))
						output[i-lx,j-ly] += local_value * coef
						sumg += coef
            #update output
			output[i-lx,j-ly] /= sumg

cpdef cython_gaussian_pr2(np.float32_t [:, :] input,
						np.float32_t [:, :] output,
						int lw,
						int pan,
						int lx,
						int ly,
						float sigma,
						int num_threads):

	cdef:
		int i, j, k, l, _
		np.float32_t [:,:] local_input
		float sumg, I, coef

#loop on the pixels for output imagge
	for i in range(lx,2*lx):
		for j in range(ly,2*ly):
			local_input = input[i-lw:i+lw+1, j-lw:j+lw+1]
			sumg = 0
			I = local_input[lw,lw]
            # Within the window for 1 pixel
            #loop over rows
			for k in range(pan):
				coef = <float> ((k-lw)*(k-lw))
				coef = coef * (-.5/sigma)
				coef = exp(coef)
				output[i-lx,j-ly] += local_input[k,lw] * coef
				sumg += coef
				for l in range(1, lw+1, 8):
                    #prange for right side of the line
					for _ in prange(l,l+8, nogil=True, schedule = 'static', chunksize =1, num_threads=num_threads):
						output[i-lx,j-ly] += local_input[k,lw+_] * exp(-.5/sigma*(<float> ((k-lw)*(k-lw)+_*_)))
						sumg += exp(-.5/sigma*(<float> ((k-lw)*(k-lw)+_*_)))
				for l in range(0, lw, 8):
                    #prange for left side of the line
					for _ in prange(l,l+8, nogil=True, schedule = 'static', chunksize =1, num_threads=num_threads):
						output[i-lx,j-ly] += local_input[k,_] * exp(-.5/sigma*(<float> ((k-lw)*(k-lw)+(_-lw)*(_-lw))))
						sumg +=exp(-.5/sigma*(<float> ((k-lw)*(k-lw)+(_-lw)*(_-lw))))
            #update output
			output[i-lx,j-ly] /= sumg


cpdef cython_bilateral_pr(np.float32_t [:, :] input,
						np.float32_t [:, :] output,
						int lw,
						int pan,
						int lx,
						int ly,
						float sigma,
						int num_threads):

	cdef:
		int i, j, k, l, _
		np.float32_t [:,:] local_input
		float sumg, I, coef, local_output

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
				coef = coef * (-.5/sigma)
				coef = exp(coef)
				local_output += local_input[k,lw] * coef
				sumg += coef
				for l in range(1, lw+1, 8):
					for _ in range(l,l+8):
						local_output += local_input[k,lw+_] * exp(-.5/sigma*((local_input[k,lw+_]-I)*(local_input[k,lw+_]-I)+<float> ((k-lw)*(k-lw)+_*_)))
						sumg += exp(-.5/sigma*((local_input[k,lw+_]-I)*(local_input[k,lw+_]-I)+<float> ((k-lw)*(k-lw)+_*_)))
				for l in range(0, lw, 8):
					for _ in range(l,l+8):
						local_output += local_input[k,_] * exp(-.5/sigma*(<float> ((local_input[k,_]-I)*(local_input[k,_]-I)+(k-lw)*(k-lw)+(_-lw)*(_-lw))))
						sumg +=exp(-.5/sigma*(<float> ((local_input[k,_]-I)*(local_input[k,_]-I)+(k-lw)*(k-lw)+(_-lw)*(_-lw))))
            #update output
			output[i-lx,j-ly] = local_output / sumg

cpdef cython_gaussian_pr3(np.float32_t [:, :] input,
						np.float32_t [:, :] output,
						int lw,
						int pan,
						int lx,
						int ly,
						float sigma,
						int num_threads):

	cdef:
		int i, j, k, l, _
		np.float32_t [:,:] local_input
		float sumg, I, coef, local_output

#loop on the pixels for output imagge
	for i in range(lx,2*lx):#int(1.1*lx)):
		for j in range(ly,2*ly):#int(1.1*ly)):
			local_input = input[i-lw:i+lw+1, j-lw:j+lw+1]
			sumg = 0.
			local_output = 0.
			I = local_input[lw,lw]
			#prange
            # Within the window for 1 pixel
            #loop over rows
			for k in prange(pan, nogil=True, schedule = 'static', chunksize =1, num_threads=num_threads):
				coef = <float> ((k-lw)*(k-lw))
				coef = coef * (-.5/sigma)
				coef = exp(coef)
				local_output += local_input[k,lw] * coef
				sumg += coef
				for l in range(1, lw+1, 8):
					for _ in range(l,l+8):
						local_output += local_input[k,lw+_] * exp(-.5/sigma*(<float> ((k-lw)*(k-lw)+_*_)))
						sumg += exp(-.5/sigma*(<float> ((k-lw)*(k-lw)+_*_)))
				for l in range(0, lw, 8):
					for _ in range(l,l+8):
						local_output += local_input[k,_] * exp(-.5/sigma*(<float> ((k-lw)*(k-lw)+(_-lw)*(_-lw))))
						sumg +=exp(-.5/sigma*(<float> ((k-lw)*(k-lw)+(_-lw)*(_-lw))))
            #update output
			output[i-lx,j-ly] = local_output / sumg
