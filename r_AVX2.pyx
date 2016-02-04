import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp
cimport AVX
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)

cpdef r_AVX2(float sigma_s,
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
		np.float32_t [:] output_array, coef_array
		float sumg, coef, I
		AVX.float8 AVX_local_input, AVX_x, AVX_y, AVX_gauss_coef, AVX_coef, AVX_ouput, AVX_I
		float val
		int pan = 2 * lw + 1

#useful avg within gaussian coefficients
	AVX_gauss_coef = AVX.float_to_float8(-.5/sigma_r**2)
	#loop over points in output
	for i in range(lx,2*lx):
		for j in range(ly,2*ly):
			local_input = input[i-lw:i+lw+1, j-lw:j+lw+1]
			sumg = 0.
			val = 0.
			output_array = np.zeros(8, dtype= np.float32)
			coef_array = np.zeros(8, dtype= np.float32)
			I = local_input[lw,lw]
			AVX_I =  AVX.float_to_float8(<float> I)
			#work on a window for one pixel output
			#loop over rows
			for k in prange(pan, nogil=True, schedule = 'static', chunksize =1, num_threads=num_threads):
				AVX_y =  AVX.float_to_float8(<float> (k-lw))
				AVX_I =  AVX.float_to_float8(<float> (k-lw))
				coef = <float> ((k-lw)*(k-lw)) + ((local_input[k,lw]-I)*(local_input[k,lw]-I))
				coef = coef * (-.5/sigma_r**2)
				# coef = exp(coef)
				val += local_input[k,lw] * coef
				sumg += coef
				# loop over columns
				for l in range(1, lw+1, 8):
					# avx for local input data
					AVX_local_input = AVX.make_float8(local_input[k,lw+l], local_input[k,lw+l+1], local_input[k,lw+l+2], local_input[k,lw+l+3],
					local_input[k,lw+l+4], local_input[k,lw+l+5],local_input[k,lw+l+6], local_input[k,lw+l+7])
					# avx for x coordonate
					AVX_x = AVX.make_float8(<float> l, <float> l+1, <float> l+2, <float> l+3, <float> l+4, <float> l+5, <float> l+6, <float> l+7)
					# avx for coefficients
					AVX_coef = AVX.sub(AVX_local_input, AVX_I)
					AVX_coef = AVX.fmadd(AVX_coef, AVX_coef, AVX.mul(AVX_x, AVX_x))
					AVX_coef = AVX.add(AVX.mul(AVX_y, AVX_y), AVX_coef)
					AVX_coef = AVX.mul(AVX_coef, AVX_gauss_coef)
					# AVX_coef = AVX.exp(AVX_coef)
					AVX_ouput = AVX.mul(AVX_local_input, AVX_coef)
					for _ in range(8):
						output_array[_] = <np.float32_t> (<np.float32_t *> &AVX_ouput)[_]
						val += output_array[_]
						coef_array[_] = <np.float32_t> (<np.float32_t *> &AVX_coef)[_]
						sumg += coef_array[_]
				for l in range(0, lw, 8):
					AVX_local_input = AVX.make_float8(local_input[k,l],local_input[k,l+1],local_input[k,l+2],local_input[k,l+3],local_input[k,l+4],
					local_input[k,l+5],local_input[k,l+6],local_input[k,l+7])
					AVX_x = AVX.make_float8(<float> l-lw,<float> l+1-lw,<float> l+2-lw,<float> l+3-lw,<float> l+4-lw,<float> l+5-lw,
					<float> l+6-lw,<float> l+7-lw)
					AVX_coef = AVX.sub(AVX_local_input, AVX_I)
					AVX_coef = AVX.fmadd(AVX_coef, AVX_coef, AVX.mul(AVX_x, AVX_x))
					AVX_coef = AVX.add(AVX.mul(AVX_y, AVX_y), AVX_coef)
					AVX_coef = AVX.mul(AVX_coef, AVX_gauss_coef)
					# AVX_coef = AVX.exp(AVX_coef)
					AVX_ouput = AVX.mul(AVX_local_input, AVX_coef)
					for _ in range(8):
						output_array[_] = <np.float32_t> (<np.float32_t *> &AVX_ouput)[_]
						val += output_array[_]
						coef_array[_] = <np.float32_t> (<np.float32_t *> &AVX_coef)[_]
						sumg += coef_array[_]
			#update output
			output[i-lx,j-ly] = val/sumg
