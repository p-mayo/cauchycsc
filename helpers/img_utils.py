# Python code for basic functions for image processing.

import numpy as np 			# To do the math
import cv2
import os
import random 
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import normalize
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sys import platform

from helpers import progress_track as pt

if platform == 'linux':
	plt.switch_backend('agg')

cdict1 = {'red':   ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.1),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 1.0),
                   (0.5, 0.1, 0.0),
                   (1.0, 0.0, 0.0))
        }

#def get_patches(img, patch_size, stride='full'):
#	yi = np.array([0])
#	[M, N] = img.shape
#	if stride == 'full':
#		stride = patch_size
#	px = 0
#	py = 0
#	atom_size = np.prod(patch_size)
#	while px + patch_size[0] <= M:
#		py = 0
#		while py + patch_size[1] <= N:
#			#yi_aux = img[px:px+patch_size[0],py:py+patch_size[1]]
#			#print("px = %d, endx = %d, py = %d endy = %d , shape=[%d, %d]" % (px, px+stride[0], py, py+stride[1],yi_aux.shape[0], yi_aux.shape[1]) )
#			yi_aux = img[px:px+patch_size[0],py:py+patch_size[1]].reshape([atom_size,1])
#			if yi.shape[0] == 1:
#				yi = yi_aux
#			else:
#				yi = np.append(yi, yi_aux, axis=1)
#			py = py + stride[1]
#		px = px + stride[0]
#		print(".", end="")
#	print(" end")
#	return yi.astype('float64')

blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)

def get_patches(img, patch_size, stride=1):
	yi = extract_patches_2d(img, patch_size)
	#print(yi.shape)
	if stride !=1:
		yi_aux = np.array([0])
		for i in range(0,yi.shape[0], (img.shape[1]-stride + 1)*stride):
			for j in range(int(img.shape[1]/stride)):
				aux = np.array([yi[i + j*stride]])
				#print("%d, %d "% (np.count_nonzero(aux), np.prod(aux.shape)))
				if (np.count_nonzero(aux)>0):
					if np.size(yi_aux.shape) == 1 :
						yi_aux = aux
					else:
						yi_aux = np.vstack([yi_aux, aux])
		return yi_aux.astype(np.float64)
	return yi.astype(np.float64)

def add_noise(img, mean, sd, seed):
	random.seed(seed)
	noise = np.random.normal(mean,sd,img.shape)
	return img + noise

def normalize(img):
	new_img = img - np.mean(img, axis=0)
	new_img = new_img / np.std(new_img, axis=0)
	return new_img

def remove_elements(img, rate, seed):
	# Cheching the rate is less than 100%
	random.seed(seed)
	mask = np.ones(img.shape)
	if rate < 1:
		# Getting total of pixels to be removed
		elements = np.product(img.shape)*rate
		# removed will contain the set of points that have already been set to 0 within the image
		removed = []
		i = 0
		rangeX = (0, img.shape[0])
		rangeY = (0, img.shape[1])
		while i < elements:
			x = random.randrange(*rangeX)
			y = random.randrange(*rangeY)
			if (x,y) in removed: 
				continue
			mask[x, y] = 0.
			removed.append((x,y))
			i += 1
	new_img = np.multiply(img, mask)
	return [new_img, mask]

def preprocess_img(img_path, intensity_norm=True):
	if img_path.endswith("pickle"):
		img = pt.load_progress(img_path)
		img = img.toarray()
		#print(img.shape)
	else:
		img = cv2.imread(img_path,0).astype(np.float64)
	if intensity_norm:
		img = img/np.max(img)
	return img

def reconstruct_patches(img_patch, patches_x, patches_y):
	patch_size = int(img_patch.shape[0]**0.5)
	img_reconstructed = np.zeros([patches_x*patch_size, patches_y*patch_size])
	atom = 0
	for i in range(patches_x):
		for j in range(patches_y):
			img_reconstructed[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size] = \
				img_patch[:,atom].reshape([patch_size, patch_size])
			atom = atom + 1
	return img_reconstructed

def visualize(dictionary, patch_size=None, title="Dictionary Learned", img_path=""):
	n_components = dictionary.shape[0]
	if patch_size == None:
		patch_size = int(np.sqrt(dictionary.shape[1]))
		patch_size = (patch_size, patch_size)

	[patches_x, patches_y] = get_factors(n_components)
	plt.figure(1, figsize=(18, 9))
	for i, comp in enumerate(dictionary[:n_components]):
	    plt.subplot(patches_x, patches_y, i + 1)
	    #plt.subplot(patches_x, patches_y, i + 1).set_title(i+1)
	    plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
	               interpolation='nearest')
	    plt.xticks(())
	    plt.yticks(())
	plt.suptitle(title, fontsize=18)
	plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.08)
	if img_path != "":
		plt.figure(1).savefig(img_path, bbox_inches='tight')
	else:
		plt.show()

def get_factors(n):
	factors = [1]
	for i in range(2, n+1):
		if (n % i) == 0:
			factors.append(i)
	tot_factors = np.size(factors)
	if tot_factors > 2:
		middle = int(tot_factors/2)
		if (tot_factors % 2 )== 0:
			return [factors[middle - 1], factors[middle]]
		else: 
			return [factors[middle], factors[middle]]
	else:
		return get_factors(n+1)

# maxVal should be 1 if the data is double and 255 for 8-bit unsigned integer
def get_psnr(orig_im, estim_im, maxval=1.):
	orig_im = orig_im[:]
	estim_im = estim_im[:]
	mse_val = (1/orig_im.size) * np.sum((orig_im - estim_im)**2) 
	psnr = 10 * np.log10(maxval**2/mse_val)
	return psnr

def get_avg_psnr(orig_imgs, estim_imgs, maxVal=1.):
	T = orig_imgs.shape[0]
	avg = 0
	for t in range(T):
		avg += get_psnr(orig_imgs[t], estim_imgs[t], maxVal)
	avg /= T
	return avg

def patch2img(patches, patch_size,im_size):
	inds = np.reshape(np.r_[0:np.prod(im_size)], im_size)
	im = np.zeros([np.prod(im_size), 1])
	#inds = get_patches(inds, patch_size, 1, True)
	inds = extract_patches_2d(inds, patch_size)
	inds = inds.reshape(inds.shape[0], -1).T
	for i in range(im.shape[0]):
		[row, col] = np.where(inds==i)
		im[i] = np.mean(patches[row, col])
	im = np.reshape(im, im_size)
	return im

def compare_img(original, corrupted, reconstructed, title, path=""):
	"""Helper function to display denoising"""
	plt.figure(1, figsize=(7.5, 3.3))
	plt.subplot(1, 3, 1)
	plt.title('Original')
	plt.imshow(original,cmap='gray')
	plt.xticks(())
	plt.yticks(())
	
	plt.subplot(1, 3, 2)
	psnr = get_psnr(original, corrupted)
	plt.title('Corrupted (PSNR=%0.2f)' % psnr )
	plt.imshow(corrupted, cmap='gray')
	plt.xticks(())
	plt.yticks(())

	plt.subplot(1, 3, 3)
	psnr = get_psnr(original, reconstructed)
	plt.title('Reconstructed (PSNR=%0.2f)' % psnr )
	plt.imshow(reconstructed, cmap='gray')
	plt.xticks(())
	plt.yticks(())

	plt.suptitle(title, size=16)
	plt.subplots_adjust(0.02, 0.02, 0.98, 0.79, 0.02, 0.2)
	if path != "":
		plt.figure(1).savefig(path, bbox_inches='tight')
	else:
		plt.show()


def show_imgs(imgs, title, path="", figsize=(18,9)):
	"""Helper function to display denoising"""
	total = imgs.shape[0]
	if total < 6:
		rows = 1
		cols = total
		fig, axes = plt.subplots(rows, cols, figsize=figsize)
		if total == 1:
			axes.imshow(imgs[0], cmap='gray', interpolation='nearest')
			axes.set_xticks(())
			axes.set_yticks(())
		else:
			for j in range(cols):
				axes[j].imshow(imgs[j], cmap='gray', interpolation='nearest')
				axes[j].set_xticks(())
				axes[j].set_yticks(())
	else:
		[rows, cols] = get_factors(total)
		fig, axes = plt.subplots(rows, cols, figsize=figsize)
		idx = 0
		#print('r = %d, c = %d' % (rows, cols))
		for i in range(rows):
			for j in range(cols):
				# print('i = %d, j= %d' % (i,j))
				axes[i,j].set_xticks(())
				axes[i,j].set_yticks(())
				if idx < total:
					axes[i,j].imshow(imgs[idx], cmap='gray', interpolation='nearest')
				idx += 1
		#plt.tight_layout()
	plt.suptitle(title, fontsize=28)
	#plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.08)
	if path != "":
		plt.savefig(path,  bbox_inches='tight')
	else:
		plt.show()
	plt.close()

def save_imgs(imgs, vmin=None, vmax=None, cmap='gray', showbar=False, path="", pattern="img"):
	if cmap == 'bluered':
		cmap = blue_red1
	if vmin == None:
		vmin = imgs.min()
	if vmax == None:
		vmax = imgs.max()

	for i in range(len(imgs)):
		fig, ax = plt.subplots()
		pcm = ax.imshow(imgs[i], vmin=vmin, vmax=vmax, cmap=cmap)
		ax.set_xticks([])
		ax.set_yticks([])
		if showbar:
			fig.colorbar(pcm,ax=ax)
		if path != "":
			plt.savefig(os.path.join(path, "%s_%d.png" % (pattern, i)),  bbox_inches='tight')
		else:
			plt.show()
		plt.close()

def get_fm_colourmap():
	return blue_red1