# Python file to perform Convolutional Sparse Coding and Convolutional
# Dictionary Learning using different thresholding algorithms

# 1. Read dataset
# 2. Pre-processing (zero mean, unit variance -- optional)
#    a) If ICT was chosen, estimate the parameter gamma from the data
# 3. Alternating:
#    a) Learn coefficients
#    b) Learn filters
# 4. Get Histogram of coefficients
#    a) Estimate the parameter gamma from them


# Datasets can be:
# a) Lena
# b) Phantom
# c) A few MRI slices (random selection)
# d) US
# e) MNIST
# f) Faces
# g) CIFAR 10

# All imports go here
import argparse 
import random
import sys

from autograd import numpy as np
from matplotlib import pyplot as plt
from sparselandtools.dictionaries import DCTDictionary

from helpers import datasets as ds
from csc.utils import dim
from stats import cauchy
from csc import cauchy_csc as ccsc

from helpers.general import str2bool
from helpers import progress_track as pt

# Constants -- Hence these do not depend on the experiment

# SC_OPT = "thresholding"
# Handles inputs and call the main functions to learn the coefficients [and
# dictionary]:
# Arguments:
#	base_path:			path were results will be saved. Further directories will
#						be created if needed
# 	dataset_path:		location of data to process (it could also say purely MNIST)
#	filter_size:		size of one side of the filters
# 	fixed:				whether the dictionary to use will be fixed (DCT)
#						or it will be learnt. It can also be learned using DCT as
#						starting point. Values: 
#						1 - Fixed, no learning
#						2 - Learned, using random intialisation
#						3 - Learned, using DCT as initial filter bank
# 
#	learning_rate:		two dimensional array, one learning rate for the
#						learning of the dictionary and another for the 
#						coefficients
# 	lmbda:				trade-off parameter -- not necessarily required for ICT
# 	n_filters:			number of filters to learn
#	n_samples:			number of samples of input data to use
#	param:				gamma for ICT
#						delta for ILT
#	prior:				for the thresholding algorithm to use. 
#						It could be laplace, cauchy, log, hard, etc
# OUTPUT:
#	z:					the learned filters
#	f:					the final dictionary used
def main(dataset_path, n_samples, n_filters, filter_size, fixed, base_path,
			prior="cauchy", lmbda=1., param=0.001, learning_rate=[0.2,0.015], 
			max_inner=5, max_outer=10, sc_opt='thresholding', seed=None):
	# Results stuff

	# Reading dataset
	data = ds.get_nd_dataset(dataset_path, n_samples)

	if n_samples != data.shape[0]:
		n_samples = data.shape[0]

	# Pre-processing (zero-mean, unit variance)
	data = (data - np.mean(data)) #/np.std(data)

	# Dealind with the shapes of any number of dimensions
	data_dim = dim(data)
	data_shape = np.array(data.shape[1:]) 

	filter_shape = [n_filters] + data_dim*[filter_size]
	featuremap_shape = [n_samples, n_filters] + list(data_shape - filter_size + 1)

	# Variables initialization
	np.random.seed(seed) # Some integer or None
	
	# Initial filter bank -- normal distribution
	if type(fixed) == str:
		f0 = pt.load_progress(fixed)
		fixed = True
	else:
		if fixed == 2:
			# Random initialisation
			dct_start = False
			f0 = np.random.normal(size=filter_shape)
		else:
			# DCT as initial dictionary
			dct_start = True
			f0 = DCTDictionary(filter_size, int(np.sqrt(n_filters)))
			f0 = f0.matrix
			f0 = f0.T.reshape(filter_shape)
		if fixed == 1:
			# If fixed == 1, it means DCT would be used
			fixed = True
		else:
			# The dictionary would be learnt for 2 and 3, regardless its initial shape
			fixed = False

	# Size of feature maps follows the convolution 'rule'
	z0 = np.zeros(featuremap_shape)
	
	if (prior == "cauchy") and (param == 0.):
		param = cauchy.estimate_gamma(data.ravel())
	elif (prior == "log") and (param == 0.): #For log thresholding this is delta
		param = 0.001

	[f, z, cost_hist] = ccsc.learn_d_z(f0, z0, data, 
		max_outer, param, learning_rate=learning_rate, lmbda=lmbda, prior=prior, 
		max_inner=max_inner, path=base_path, sc_opt=sc_opt,	fixed=fixed, 
		plot_hist=False, dataset=dataset_path, dct_start=dct_start, seed=seed)

	

# python -m csctesting.csc_thresholding -dp mnist -pr cauchy -bp C:\phd\results\csc_thresholding\mnist -ns 100 -fx 2
# python -m csctesting.csc_thresholding -dp C:\phd\data\all -pr cauchy -bp C:\phd\results\csc_thresholding\faces -fx 2
# python -m csctesting.csc_thresholding -dp mnist -pr cauchy -bp C:\phd\results\csc_thresholding_fixed\mnist -ns 100 -fx 1
# python -m csctesting.csc_thresholding -dp C:\phd\data\all -pr cauchy -bp C:\phd\results\csc_thresholding_fixed\faces -fx 1
# python -m csctesting.csc_thresholding -dp mnist -pr cauchy -bp C:\phd\results\csc_thresholding_dct_learned\mnist -ns 100
# python -m csctesting.csc_thresholding -dp C:\phd\data\all -pr cauchy -bp C:\phd\results\csc_thresholding_dct_learned\faces
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Learning filters and feature maps for CSC')
	parser.add_argument('-dp','--datasetpath', help='', type=str)
	parser.add_argument('-ns','--numsamples', help='', type=int, default=0)
	parser.add_argument('-nf','--numfilters', help='', type=int, default=64)
	parser.add_argument('-fs','--filtersize', help='', type=int, default=7)
	parser.add_argument('-fx','--fixed', help='', type=int, default=3)
	parser.add_argument('-bp','--basepath', help='', type=str)
	parser.add_argument('-pr','--prior', help='', type=str, default="cauchy")
	parser.add_argument('-lm','--lambda', help='', type=float, default=1.)
	parser.add_argument('-pm','--param', help='', type=float, default=0.)
	parser.add_argument('-lrd','--learningrated', help='', type=float, default=0.2)
	parser.add_argument('-lrz','--learningratez', help='', type=float, default=0.015)
	parser.add_argument('-mi','--maxinner', help='', type=int, default=30)
	parser.add_argument('-mo','--maxouter', help='', type=int, default=10)
	parser.add_argument('-sc','--scopt', help='', type=str, default='thresholding')
	parser.add_argument('-md','--mode', help='', type=str, default='convolutional')
	parser.add_argument('-sd','--seed', help='', type=int, default=0)
	args = vars(parser.parse_args())

	dataset_path     = args['datasetpath'] 
	n_samples        = args['numsamples']
	n_filters        = args['numfilters'] 
	filter_size      = args['filtersize'] 
	fixed            = args['fixed']
	base_path        = args['basepath']
	prior            = args['prior'] 
	lmbda            = args['lambda'] 
	param            = args['param']
	learning_rate_d  = args['learningrated'] 
	learning_rate_z  = args['learningratez'] 
	max_inner        = args['maxinner'] 
	max_outer        = args['maxouter']
	sc_opt        	 = args['scopt']
	mode 			 = args['mode']
	seed 			 = args['seed']

	if seed == -1:
		seed = random.randrange(2**32 -1)

	if mode == "convolutional":
		main(dataset_path, n_samples, n_filters, filter_size, fixed, base_path, prior, lmbda, 
				param, [learning_rate_d, learning_rate_z], max_inner, max_outer, sc_opt, seed)