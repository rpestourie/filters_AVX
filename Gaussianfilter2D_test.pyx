from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
# ------------------------------------------------------------
# import modules for cython and AVX
# ------------------------------------------------------------
cimport cython
# cimport AVX
from cython.parallel import prange
cimport numpy as np

cpdef _AVX_cython_convolution(int lw,
							  int lx,
							  int ly,
							  np.float32_t [:, :] image_in,
							  np.float32_t [:,:] image_out,
							  np.float32_t [:,:] kernel):

	cdef:
		int i,j , i_local
		AVX.float8 local_input_AVX

	# can optimize with chunksize too
	for i in prange(0,lx , nogil= True, schedule = 'static', num_threads = 2):
		for j in range(0,ly):
			local_input = image[i : i + 2*self.lw + 1, j: j + 2*self.lw + 1]			
			# summation of the two (2*lw+1 , 2*lw+1) done with AVX
			for i_local in range(0, local_input.shape[0] + 1):
				# sum the left part
				# store the 8 adjacent values into one AVX
				local_input_AVX_left = AVX.make_float(local_input[i_local, 7],
												 local_input[i_local, 6],
												 local_input[i_local, 5]
												 local_input[i_local, 4]
												 local_input[i_local, 3]
												 local_input[i_local, 2]
												 local_input[i_local, 1]
												 local_input[i_local, 0])
				kernel =  AVX.make_float(kernel[i,7],
										 kernel[i,6],
										 kernel[i,5],
										 kernel[i,4],
										 kernel[i,3],
										 kernel[i,2],
										 kernel[i,1],
										 kernel[i,0])														)

			image_out[i, j] = image_in[i, j]

	return image_out

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
		image = np.array(image, np.float32)
		image_out = np.zeros((lx,ly), dtype = np.float32)
		_AVX_cython_convolution(self.lw, lx, ly, image, image_out, self._kernel)

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







