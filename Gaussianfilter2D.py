from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter



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

	def _other(self, local_input):

		toplot = local_input*0.0
		sumg = 0
		for k in range(self.pan):
			   gaussianc = np.exp(-0.5 / self.sigma**2
			    			*np.linalg.norm([k-self.lw,0])**2)
			   sumg += gaussianc
			   toplot[k,self.lw] = local_input[k,self.lw] * gaussianc
			   for l in range(1, self.lw+1, 8):
			       toplot[k,self.lw+l:self.lw+l+8] = [local_input[k,self.lw+_]*np.exp(-.5/self.sigma*np.linalg.norm([k-self.lw,_])**2) for _ in range(l,l+8)]
			       gaussianl = [np.exp(-.5/self.sigma*np.linalg.norm([k-self.lw,_])**2) for _ in range(l,l+8)]
			       sumg += np.sum(gaussianl)
			   for l in range(0, self.lw, 8):
			       toplot[k,l:l+8] = [local_input[k,_]*np.exp(-.5/self.sigma*np.linalg.norm([k-self.lw,_-self.lw])**2) for _ in range(l,l+8)]
			       gaussianr = [np.exp(-.5/self.sigma*np.linalg.norm([k-self.lw,_-self.lw])**2) for _ in range(l,l+8)]
			       sumg += np.sum(gaussianr)
		toplot /= sumg
		self.image_ = toplot

		# run the filter with scipy to get error and run time difference
		self.filter_scipy(local_input)
		self.error_ = np.linalg.norm(self.image_benchmark_-self.image_)

		return self.image_


	def filter_python(self,f):


		start = time.time()
		self.image_ = f * 0.0
		lx, ly = f.shape
		# create the gaussian filter kernel
		self._kernel = self.kernel_

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



		# convolution with the gaussian kernel for filtering
		self.image_= self._python_convolution(lx,ly,image)


		self.run_time_ = time.time() - start

		# run the filter with scipy to get error and run time difference
		self.filter_scipy(f)
		self.error_ = np.linalg.norm(self.image_benchmark_-self.image_)

		return self







