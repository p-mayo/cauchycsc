# Python script to test the lap Distribution and it's optimization
# using autograd.
import os

import autograd
import autograd.numpy as np
import autograd.scipy.signal

from scipy import optimize
from autograd import grad
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize

#from csc.cauchy_csc import learn_z

convolve = autograd.scipy.signal.convolve


# Plotting utils...
def plot_signal(s, T=[-1], K=[-1], title=None, path="", pattern=""):
	if np.size(s.shape) < 3:
		s = np.array([s])
	if T[0]==-1:
		T = range(s.shape[0])
	if K[0]==-1:
		K= range(s.shape[1])
	#print("T = %d, K = %d" % (size(T), size(K)))
	for t in T:
		fig, ax = plt.subplots()
		ax.axhline(y=0, color='k')
		ax.axvline(x=0, color='k')
		for k in K:
			ax.plot(s[t, k, :])
			if title != None:
				ax.set_title("%s\nT %d, K %d" % (title, t, k))
		if path != "":
			if (not path.endswith('.png')):
				img_path = os.path.join(path, "trial_%d_%s"% (t, pattern))
			else:
				img_path = path
			plt.savefig(img_path, bbox_inches='tight')
		else:
			plt.show()
		plt.close()


def compare_signals(s1, s2, T=[-1], K=[-1], title=None, same_plot=False, same_scale=True, path = ""):
	#if s1.shape == s2.shape:
	if np.size(s1.shape) < 3:
		s1 = np.array([s1])
		s2 = np.array([s2])
	if K[0]==-1:
		K = range(s1.shape[2])
	if T[0]==-1:
		T = range(s1.shape[0])
	if same_scale:
		y_max = np.max([s1.max(), s1.max()])*1.1
		y_min = np.min([s1.min(), s1.min()])*1.1
	print('T = %s, K = %s' % (str(T), str(K)))
	color_p='r'
	color_s='b'
	figsize=(20,7)
	print(path)
	for t in T:
		for k in K:
			#plt.figure()
			if same_plot:
				fig, axs = plt.subplots(1, 2, figsize=figsize)
				axs[0].axhline(y=0, color='k')
				axs[0].plot(s1[t, k, :], color=color_p, label='Original')
				axs[0].plot(s2[t, k, :], color=color_s, label='Estimated')
				axs[0].set_title('Compared signals')
				axs[0].axis(ymin=y_min, ymax=y_max)
				axs[0].legend(['Original', 'Estimated'])
				axs[1].axhline(y=0, color='k')
				axs[1].plot(s1[t,k, :] - s2[t,k,:], color=color_p)
				if same_scale:
					axs[1].axis(ymin=y_min, ymax=y_max)
				axs[1].set_title('Difference')
			else:
				fig, axs = plt.subplots(1, 3, figsize=figsize)
				axs[0].axhline(y=0, color='k')
				axs[0].plot(s1[t,k,:], color=color_p)
				axs[0].axis(ymin=y_min, ymax=y_max)
				axs[0].set_title('Original Signal')
				axs[1].axhline(y=0, color='k')
				axs[1].plot(s2[t,k,:], color=color_p)
				axs[1].axis(ymin=y_min, ymax=y_max)
				axs[1].set_title('Estimated Signal')
				axs[2].axhline(y=0, color='k')
				axs[2].plot(s1[t,k,:] - s2[t,k,:], color=color_p)
				if same_scale:
					axs[2].axis(ymin=y_min, ymax=y_max)
				axs[2].set_title('Difference')				
			if title != None:
				fig.suptitle("%s\nT %d, K %d" % (title, t, k))
			if path != "":
				plt.savefig("%s_t_%d_k_%d.png" % (path, t, k),  bbox_inches='tight')
			else:
				plt.show()
			plt.close()


def plot_multiple_singals(signals, T=[-1], K=[-1], title=None, path = "", th=0.0):
	# signals should contain as first entry the dictionary atoms, the second is the reconstruction
	# and the third correspond to the coefficient vector. T specifies the trials to plot and K the specific atoms
	# to plot. 
	#if s1.shape == s2.shape:
	dictionary = signals[0]
	recons = signals[1]
	coeffs = signals[2]
	if K[0]==-1:
		K = range(dictionary.shape[0])
	if T[0]==-1:
		T = range(recons.shape[0])
	tot_plots = np.size(T) + 2
	cols = 4
	rows = int(np.ceil(tot_plots/4))
	figsize=(20,7)
	fig, axs = plt.subplots(rows, cols, figsize=figsize)
	#fig, axs = plt.subplots(rows, cols)

	# Plotting dictionary
	for k in K:
		axs[0,0].plot(dictionary[k,:])
	axs[0,0].set_title('Dictionary Atoms')
	for t in T:
		axs[0,1].plot(recons[t,:])
	axs[0,1].set_title('Reconstructed Signals')
	t=0
	curr_col = 2
	curr_row = 0
	y_max = np.abs(coeffs).max()*1.1
	if y_max < th:
		y_max = th*1.2
	elif y_max == 0:
		y_max = 1.
	y_min = -y_max
	#print("rows = %s, cols = %s" % (str(rows), str(cols)))
	while curr_row < rows:
		while curr_col < cols:
			#plt.figure()
			for k in K:
				axs[curr_row, curr_col].plot(coeffs[t,k,:])
			axs[curr_row, curr_col].set_title('Coefficients of trial %s' % (str(t+1)))
			axs[curr_row, curr_col].axis(ymin=y_min, ymax=y_max)
			if th != 0.:
				axs[curr_row, curr_col].axhline(y=th, color='r', linestyle='dashed')
				axs[curr_row, curr_col].axhline(y=-th, color='r', linestyle='dashed')
			t += 1
			if t >= recons.shape[0]:
				curr_row = rows
				curr_col = cols
			curr_col += 1
		curr_col = 0
		curr_row += 1
	if title != None:
		fig.suptitle(title)
	if path != "":
		plt.savefig(path,  bbox_inches='tight')
	else:
		plt.show()
	plt.close()

def estimate_y(z, d):
	# Autograd doesn't support assignment into arrays, so to use autograd 
	# you have to rewrite your code to avoid it...
	#dConvZ = np.zeros([P,T]).astype(np.float64)
	#for t in range(T):
	#	for k in range(K):
	#		dConvZ[:,t] += convolve(z[t,:,k],d[:,k])
	# Using CONVOLVE function instead:
	# axes indicates where the information of the signal to convolve is.
	# 	   In this case, for z, the first axis specifies the image it belongs to
	#      whilst the third one indicates the "channel", therefore, the signal
	#      to convolve is in the second axis. For d, the second axis specifies
	#      the channel, so the first one contains the signal.
	# dot  indicates the channels of the signals to convolve, so the result of the
	#      convolutions will be added along this axis.
	#dConvZ = convolve(z, d, axes=([1],[0]), dot_axes=([2],[1]), mode='full')
	# Checking the dimensions to work with
	# 1D
	if dim(d) == 1: 
		dConvZ = convolve(z, d, axes=([2],[1]), dot_axes=([1],[0]), mode='full')
	elif dim(d) == 2: 
		dConvZ = convolve(z, d, axes=([2,3],[1,2]), dot_axes=([1],[0]), mode='full')
	elif dim(d) == 3: 
		dConvZ = convolve(z, d, axes=([2,3,4],[1,2,3]), dot_axes=([1],[0]), mode='full')
	return dConvZ

def dim(arr):
	# For an array of shape [T, H, W, ...], the first number (T) corresponds to the
	# number of samples, hence the dimension of the data worked with is obtained based
	# on the rest of the numbers in that shape
	return np.size(arr.shape) - 1

def get_residual(y, y_hat):
	return y - y_hat

# Get a set of feature maps and obtains the average of non_zero coefficients per
# feature map
def sparsity_metric(z, mode="fm"):
	num_coeffs = np.count_nonzero(z,axis=1)
	if mode == "atom" :
		return np.mean(num_coeffs, axis=0)
	if mode == "l0" :
		return np.sum(num_coeffs)
	else: # average per every atom	
		return np.mean(num_coeffs)
