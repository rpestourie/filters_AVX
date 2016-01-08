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

	THIS IS SOMETHING TO DO



	'''
	def __init__(self, sigma, truncate = 4.0 ):


		self.sigma = sigma
		self.truncate = truncate


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
		self._kernel *= -.5/self.sigma
		self._kernel = np.exp(self._kernel)
		self._kernel /= np.sum(self._kernel)

		return self._kernel

	def filter_scipy(self, f , order = 2):

		self.image_ = gaussian_filter(f, order)

		return self.image_
	

	def filter_python(self,f):
		# first step is to reproduce the periodic boundary conditions
		# add halo (periodic boundary condition, 
		# we could do something better but we don't have time to mess 
		#  around with indexes...)
		input = np.zeros((3*lx,3*ly))
		for i in range(3):
		    for j in range(3):
		            input[i*lx:(i+1)*lx, j*ly:(j+1)*ly] = f


		# convolution of the image with our gaussian filter
		for i in range(lx,2*lx):
		    for j in range(ly,2*ly):
		        local_input = input[i-lw:i+lw+1, j-lw:j+lw+1]
		        output_ini[i-lx,j-ly]= np.sum(local_input*gaussianfilter)






