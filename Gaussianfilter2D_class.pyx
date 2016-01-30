from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
# ------------------------------------------------------------
# import modules for cython and AVX
# ------------------------------------------------------------
cimport cython
cimport AVX
from cython.parallel import prange
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)




cpdef _testing (int lw, int lx, int ly, np.float32_t [:,:] image_in, np.float32_t [:,:] image_out,
				np.float32_t [:,:] kernel ):

	cdef:
		int i,j, i_local, k, m_8, n_elem
		np.float32_t [:,:] local_input
		np.float32_t [:] output_array_left,output_array_right, \
							output_array_top, output_array_bot
		AVX.float8 AVX_coef_left, kernel_AVX_left, local_input_AVX_left, \
							AVX_coef_right, kernel_AVX_right, local_input_AVX_right,\
							AVX_coef_top, kernel_AVX_top , local_input_AVX_top, \
							AVX_coef_bot, kernel_AVX_bot, local_input_AVX_bot
		float sumg , check_sum_manual
		int j_m

	n_elem = lw / 8


	for i in range(lx):
		for j in range(ly):
			sumg = 0.0
			local_input = image_in[i : i + 2*lw + 1, j: j + 2*lw + 1]
			# for i_local in range(local_input.shape[0]):
			for i_local in prange(local_input.shape[0], \
				nogil=True, schedule = 'static', chunksize =1, num_threads=2):	
				for m_8 in range(n_elem):
					# left part
					local_input_AVX_left = AVX.make_float8(local_input[i_local,m_8*8+7],
													 local_input[i_local,m_8*8+ 6],
													 local_input[i_local,m_8*8+ 5],
													 local_input[i_local,m_8*8+ 4],
													 local_input[i_local,m_8*8+ 3],
													 local_input[i_local,m_8*8+ 2],
													 local_input[i_local,m_8*8+ 1],
													 local_input[i_local,m_8*8+ 0])
					kernel_AVX_left =  AVX.make_float8(kernel[i_local,m_8*8+7],
											 kernel[i_local,m_8*8+6],
											 kernel[i_local,m_8*8+5],
											 kernel[i_local,m_8*8+4],
											 kernel[i_local,m_8*8+3],
											 kernel[i_local,m_8*8+2],
											 kernel[i_local,m_8*8+1],
											 kernel[i_local,m_8*8+0])
					AVX_coef_left = AVX.mul(local_input_AVX_left,kernel_AVX_left)

					# right part
					local_input_AVX_right = AVX.make_float8(local_input[i_local,lw+1+m_8*8+ 7],
													 local_input[i_local, lw+1+ m_8*8+6],
													 local_input[i_local, lw+1+ m_8*8+5],
													 local_input[i_local, lw+1+ m_8*8+4],
													 local_input[i_local, lw+1+ m_8*8+3],
													 local_input[i_local, lw+1+ m_8*8+2],
													 local_input[i_local, lw+1+ m_8*8+1],
													 local_input[i_local,lw+1+  m_8*8+0])
					kernel_AVX_right =  AVX.make_float8(kernel[i_local,lw+1+m_8*8+7],
											 kernel[i_local,lw+1+m_8*8+6],
											 kernel[i_local,lw+1+m_8*8+5],
											 kernel[i_local,lw+1+m_8*8+4],
											 kernel[i_local,lw+1+m_8*8+3],
											 kernel[i_local,lw+1+m_8*8+2],
											 kernel[i_local,lw+1+m_8*8+1],
											 kernel[i_local,lw+1+m_8*8+0])
					AVX_coef_right = AVX.mul(local_input_AVX_right,kernel_AVX_right)	

					for k in range(8):
						sumg += <np.float32_t> (<np.float32_t *> &AVX_coef_left)[k]
						sumg += <np.float32_t> (<np.float32_t *> &AVX_coef_right)[k]
				

			# top and bottom
			local_input_AVX_top = AVX.make_float8(local_input[m_8*8+ 7, lw],
											 local_input[ m_8*8+6, lw],
											 local_input[ m_8*8+5, lw],
											 local_input[ m_8*8+4, lw],
											 local_input[ m_8*8+3, lw],
											 local_input[ m_8*8+2, lw],
											 local_input[ m_8*8+1, lw],
											 local_input[ m_8*8+0, lw])
			kernel_AVX_top =  AVX.make_float8(kernel[m_8*8+7, lw],
									 kernel[m_8*8+6, lw],
									 kernel[m_8*8+5, lw],
									 kernel[m_8*8+4, lw],
									 kernel[m_8*8+3, lw],
									 kernel[m_8*8+2, lw],
									 kernel[m_8*8+1, lw],
									 kernel[m_8*8+0, lw])
			AVX_coef_top = AVX.mul(local_input_AVX_top,kernel_AVX_top)	

			local_input_AVX_bot = AVX.make_float8(local_input[lw+1+ m_8*8+7, lw],
											 local_input[ lw+1+ m_8*8+6, lw],
											 local_input[ lw+1+ m_8*8+5, lw],
											 local_input[ lw+1+ m_8*8+4, lw],
											 local_input[ lw+1+ m_8*8+3, lw],
											 local_input[ lw+1+ m_8*8+2, lw],
											 local_input[ lw+1+ m_8*8+1, lw],
											 local_input[lw+1+  m_8*8+0, lw])
			kernel_AVX_bot =  AVX.make_float8(kernel[lw+1+m_8*8+7, lw],
									 kernel[lw+1+m_8*8+6, lw],
									 kernel[lw+1+m_8*8+5, lw],
									 kernel[lw+1+m_8*8+4, lw],
									 kernel[lw+1+m_8*8+3, lw],
									 kernel[lw+1+m_8*8+2, lw],
									 kernel[lw+1+m_8*8+1, lw],
									 kernel[lw+1+ m_8*8+0, lw])
			AVX_coef_bot = AVX.mul(local_input_AVX_bot,kernel_AVX_bot)


			for k in range(8):
				sumg += <np.float32_t> (<np.float32_t *> &AVX_coef_top)[k]
				sumg += <np.float32_t> (<np.float32_t *> &AVX_coef_bot)[k]

			# middle one
			sumg += kernel[lw,lw]*local_input[lw,lw]										
			# compute the filtered image
			image_out[i, j] = sumg
	return 					

			



cdef _cython_convolution(int lw,
						int  lx,
						int  ly,
						np.float32_t [:,:] image_in,
						np.float32_t [:,:] image_out,
						np.float32_t [:,:] kernel):

	cdef:
		int i, j, i_local, j_local
		np.float32_t [:,:] local_input
		float sumg

	# convolution with the gaussian kernel for filtering
	for i in range(0 , lx ):
		for j in range(0 , ly ):
			local_input = image_in[i : i + 2* lw + 1, j: j + 2* lw + 1]
			sumg = 0.0
			for i_local in range(local_input.shape[0]):
				for j_local in range(local_input.shape[1]):
					sumg += local_input[i_local, j_local]*kernel[i_local,j_local]
			image_out[i, j] = sumg
	return 


class Gaussianfilter2D():
	'''
	2D gaussian filter on Black and White images

	The filter image satisfies the relation
	$$ C_{i,j} = \sum_{m=0}^{2 l_w} \sum_{n=0}^{2 lw} = K_{m,n} \, I_{i-lw + m, j-l_w+n}$$
	where C is the new image, K the gaussian kernel and I the original image

	Parameters

	----------

	sigma : float

		standard deviation for the gaussian filter

	truncate : float, optional

		truncate the filter at this many standard deviations
		Default is 4.0


	mode : here are the various mode supported by the class

    mode       |   Ext   |         Input          |   Ext
    -----------+---------+------------------------+---------
    'reflect'  | 3  2  1 | 1  2  3  4  5  6  7  8 | 8  7  6	
    'nearest'  | 1  1  1 | 1  2  3  4  5  6  7  8 | 8  8  8
    'constant' | 0  0  0 | 1  2  3  4  5  6  7  8 | 0  0  0
    'wrap'     | 6  7  8 | 1  2  3  4  5  6  7  8 | 1  2  3    


	Attributes

	----------   

	kernel_ : 2d-array

		this is the 2D gaussian filter kernel used in the convolution
		with the provided image

	image_benchmark_: 2d-array

		this is the image filtered with the 2D gaussian filter provided
		by the scipy library

	run_time_benchmark_: float

		this is the run time of the 2D gaussian filter provied by the
		scipy library

	image_: 2d-array

		this is the image filtered by the 2D gaussian filter implemented in 
		python

	run_time_: float

		this is the run time of the 2D gaussian filter implemented in python

	error_ : float

		this is the norm-2 error of the python function compared with the scipy
		function

	'''
	def __init__(self, sigma, truncate = 4.0 , mode = 'reflect', cval = 0.0):


		self.sigma = sigma
		self.truncate = truncate
		self.mode = mode
		self.cval = cval


		# lw is the number of adjacent pixels to consider in 1D
		# when using the filter	
		self.lw = int(truncate*sigma+0.5) 	

		# pan is the size of the 1D window used to convolute with the gaussian filter
		# it needs 16 pixels on both sides + the considered pixel in the middle
		self.pan = 2 * self.lw + 1

		# for now we take values on the right until it fills a multiple of 8
		# AVX is in x direction (in the matrix)
		self.span = (self.pan // 8 + 1) * 8

	@property
	def kernel_(self):

		'''
		this function generates the Gaussianfilter2D kernel
		'''

		# initialize the size of the gaussian kernel
		# kernel size: pan * pan (33 * 33 when truncate = 4.0 and sigma = 4.0)
		self._kernel = np.zeros((self.pan,self.pan))

		# find the distance to the center of all pixels in the kernel
		for i in range(0,self.lw+1):
			for j in range(0,self.lw+1):
		        # pixel at the center the distance is 0
				if i == 0 and j ==0:
					self._kernel[self.lw,self.lw] = 0
				# the other pixels in the kernel
				else:
					self._kernel[i+self.lw,j+self.lw] = np.linalg.norm([i,j])**2
					self._kernel[-i+self.lw,-j+self.lw] = np.linalg.norm([i,j])**2
					self._kernel[-i+self.lw,j+self.lw] = np.linalg.norm([i,j])**2
					self._kernel[i+self.lw,-j+self.lw] = np.linalg.norm([i,j])**2
		        
		# compute the gaussian kernel
		self._kernel *= -.5/self.sigma**2
		self._kernel = np.exp(self._kernel)
		self._kernel /= 2*np.pi * self.sigma**2
		self._kernel /= np.sum(self._kernel)

		return self._kernel

	def filter_scipy(self, f ):
		start = time.time()
		self.image_benchmark_ = gaussian_filter(f,self.sigma , mode = self.mode, cval = self.cval )
		self.run_time_benchmark_ = time.time() - start
		return self.image_benchmark_, self.run_time_benchmark_

	def _python_convolution(self, lx, ly, image):

		# convolution with the gaussian kernel for filtering
		for i in range(0 , lx ):
			for j in range(0 , ly ):
				local_input = image[i : i + 2*self.lw + 1, j: j + 2*self.lw + 1]
				self.image_[i , j]= np.sum(local_input*self._kernel)
		return self.image_

	def _return_AVX_cython_convolution(self, lx, ly, image):

		# convolution using the Gaussian kernel

		# initialize for cython
		image_in = np.array(image, dtype = np.float32)
		image_out = np.zeros((lx,ly), dtype = np.float32)
		kernel = np.array(self._kernel, dtype = np.float32)
		# _AVX_cython_convolution(self.lw, lx, ly, image_in, image_out, self._kernel)
		_testing(self.lw, lx, ly, image_in, image_out, kernel)
		# _cython_convolution(self.lw, lx, ly, image_in, image_out, kernel)

		self.image_ = image_out

		return self.image_

	def filter_cython(self,f):


		start = time.time()
		self.image_ = f * 0.0
		lx, ly = f.shape
		# create the gaussian filter kernel
		self._kernel = self.kernel_

		# implement the different type of method to treat the edges
		image = self._padding(f)

		# convolution with the gaussian kernel for filtering
		self.image_= self._return_AVX_cython_convolution(lx,ly,image)

		self.run_time_ = time.time() - start

		# run the filter with scipy to get error and run time difference
		self.filter_scipy(f)
		self.error_ = np.linalg.norm(self.image_benchmark_-self.image_)

		return self

	def filter_python(self,f):


		start = time.time()
		self.image_ = f * 0.0
		lx, ly = f.shape
		# create the gaussian filter kernel
		self._kernel = self.kernel_

		# implement the different type of method to treat the edges
		image = self._padding(f)

		# convolution with the gaussian kernel for filtering
		self.image_= self._python_convolution(lx,ly,image)

		self.run_time_ = time.time() - start

		# run the filter with scipy to get error and run time difference
		self.filter_scipy(f)
		self.error_ = np.linalg.norm(self.image_benchmark_-self.image_)

		return self

	def _padding(self, f):
		
		# implement the different type of method to treat the edges

		if self.mode == 'constant':

			# padding using the scipy library
			image = np.lib.pad(f , self.lw, 'constant', constant_values = self.cval)	

		elif self.mode == 'reflect':

			# padding using the scipy library
			image = np.lib.pad(f , self.lw, 'reflect')


		elif self.mode == 'wrap':

			# padding using the scipy library
			image = np.lib.pad(f , self.lw, 'wrap')

		elif self.mode == 'nearest':	

			# padding using the scipy library
			image = np.lib.pad(f , self.lw, 'edge')

		return image






