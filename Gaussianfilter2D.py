from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter



class Gaussianfilter2D():
	'''
	2D gaussian filter on Black and White images

	Parameters

	----------

	sigma : float

		standard deviation for the gaussian filter

	truncate : float, optional

		truncate the filter at this many standard deviations
		Default is 4.0


	mode : 'periodic'

    mode       |   Ext   |         Input          |   Ext
    -----------+---------+------------------------+---------
    'mirror'   | 4  3  2 | 1  2  3  4  5  6  7  8 | 7  6  5
    'reflect'  | 3  2  1 | 1  2  3  4  5  6  7  8 | 8  7  6	
    'nearest'  | 1  1  1 | 1  2  3  4  5  6  7  8 | 8  8  8
    'constant' | 0  0  0 | 1  2  3  4  5  6  7  8 | 0  0  0
    'wrap'     | 6  7  8 | 1  2  3  4  5  6  7  8 | 1  2  3    

	THIS IS SOMETHING TO DO



	'''
	def __init__(self, sigma, truncate = 4.0 , mode = 'constant', cval = 0.0):


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
		# self._kernel /= np.sum(self._kernel)

		return self._kernel

	def filter_scipy(self, f ):
		start = time.time()
		self.image_benchmark_ = gaussian_filter(f,self.sigma , mode = self.mode, cval = self.cval )
		self.run_time_benchmark_ = time.time() - start
		return self.image_benchmark_, self.run_time_benchmark_


	def filter_python(self,f):


		start = time.time()
		self.image_ = f * 0.0
		lx, ly = f.shape
		self._kernel = self.kernel_

		if self.mode == 'periodic':


			# first step is to reproduce the periodic boundary conditions
			# add halo (periodic boundary condition, 
			# we could do something better but we don't have time to mess 
			#  around with indexes...)

			image = np.zeros((3*lx,3*ly))
			for i in range(3):
			    for j in range(3):
			            image[i*lx:(i+1)*lx, j*ly:(j+1)*ly] = f


			# convolution of the image with our gaussian filter

			for i in range(lx,2*lx):
			    for j in range(ly,2*ly):
			        local_input = image[i-self.lw:i+self.lw+1, j-self.lw:j+self.lw+1]
			        self.image_[i-lx,j-ly]= np.sum(local_input*self._kernel)

		elif self.mode == 'mirror':

			image = np.zeros((3*lx,3*ly))
			# center is the image
			image[lx: 2*lx , ly : 2*ly ] = f
			# take care of the left and right sides
			for j in range(ly, 2*ly):
				for i in range(1,self.lw + 1):
					image[ lx -i, j] = image[ lx + i, j]
					image[2*lx -1 + i , j] = image[ 2*lx  -1 -i , j]
			# take care of the top and bottom
			for i in range(lx, 2*lx):
				for j in range(1, self.lw + 1 ):
					image[i, ly - j] = image[i, ly + j]
					image[i, 2*ly -1 + j] = image[i, 2*ly -1 - j]


		elif self.mode == 'constant':

			# padding using the scipy library
			image = np.lib.pad(f , self.lw, 'constant', constant_values = self.cval)

			# convolution with the gaussian kernel for filtering
			for i in range(0 , lx ):
				for j in range(0 , ly ):
					local_input = image[i : i + 2*self.lw + 1, j: j + 2*self.lw + 1]
					self.image_[i , j]= np.sum(local_input*self._kernel)					

		

		self.run_time_ = time.time() - start

		# run the filter with scipy to get error and run time difference
		self.filter_scipy(f)
		self.error_ = np.linalg.norm(self.image_benchmark_-self.image_)

		return self.image_, self.run_time_, self.error_







