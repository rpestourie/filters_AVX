import numpy as np
cimport numpy as np
cimport cython
import numpy
# cimport AVX
from cython.parallel import prange

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
		float sumg, coef, I


	for i in range(lx,2*lx):
		for j in range(ly,2*ly):
			local_input = input[i-lw:i+lw+1, j-lw:j+lw+1]
			sumg = 0
			I = local_input[lw,lw]
			for k in range(pan):
				coef = np.exp(-.5/sigma*(local_input[k,lw]-I)**2)*np.exp(-.5/sigma*np.linalg.norm([k-lw,0])**2)
				output[i-lx,j-ly] += local_input[k,lw] * coef
				sumg += coef
				for l in range(1, lw+1, 8):
					for _ in range(l,l+8):
						coef = np.exp(-.5/sigma*(local_input[k,lw+_]-I)**2) * np.exp(-.5/sigma*np.linalg.norm([k-lw,_])**2)
						output[i-lx,j-ly] += local_input[k,lw+_] * coef
						sumg += coef
				for l in range(0, lw, 8):
					for _ in range(l,l+8):
						coef = np.exp(-.5/sigma*(local_input[k,_]-I)**2)*np.exp(-.5/sigma*np.linalg.norm([k-lw,_-lw])**2)
						output[i-lx,j-ly] += local_input[k,_] * coef
						sumg += coef
			output[i-lx,j-ly] /= sumg


# small change to get to bilateral
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
		float coef, sumg, I


	for i in range(lx,2*lx):
		for j in range(ly,2*ly):
			local_input = input[i-lw:i+lw+1, j-lw:j+lw+1]
			sumg = 0
			for k in range(pan):
				coef = np.exp(-.5/sigma*np.linalg.norm([k-lw,0])**2)
				output[i-lx,j-ly] += local_input[k,lw] * coef
				sumg += coef
				for l in range(1, lw+1, 8):
					for _ in range(l,l+8):
						coef = np.exp(-.5/sigma*np.linalg.norm([k-lw,_])**2)
						output[i-lx,j-ly] += local_input[k,lw+_] * coef
						sumg += coef
				for l in range(0, lw, 8):
					for _ in range(l,l+8):
						coef = np.exp(-.5/sigma*np.linalg.norm([k-lw,_-lw])**2)
						output[i-lx,j-ly] += local_input[k,_] * coef
						sumg += coef
			output[i-lx,j-ly] /= sumg

cpdef cython_bilateral_pr(np.float32_t [:, :] input,
						np.float32_t [:, :] output,
						int lw,
						int pan,
						int lx,
						int ly,
						float sigma):

	cdef:
		int i, j, k, l, _
		np.float32_t [:,:] local_input
		float sumg, coef, I


	for i in range(lx,2*lx):
		for j in range(ly,2*ly):
			local_input = input[i-lw:i+lw+1, j-lw:j+lw+1]
			sumg = 0
			I = local_input[lw,lw]
			#prange
			for k in prange(pan, nogil=True, num_threads=2):
				coef = np.exp(-.5/sigma*(local_input[k,lw]-I)**2)*np.exp(-.5/sigma*np.linalg.norm([k-lw,0])**2)
				output[i-lx,j-ly] += local_input[k,lw] * coef
				sumg += coef
				for l in range(1, lw+1, 8):
					for _ in range(l,l+8):
						coef = np.exp(-.5/sigma*(local_input[k,lw+_]-I)**2) * np.exp(-.5/sigma*np.linalg.norm([k-lw,_])**2)
						output[i-lx,j-ly] += local_input[k,lw+_] * coef
						sumg += coef
				for l in range(0, lw, 8):
					for _ in range(l,l+8):
						coef = np.exp(-.5/sigma*(local_input[k,_]-I)**2)*np.exp(-.5/sigma*np.linalg.norm([k-lw,_-lw])**2)
						output[i-lx,j-ly] += local_input[k,_] * coef
						sumg += coef
			output[i-lx,j-ly] /= sumg
