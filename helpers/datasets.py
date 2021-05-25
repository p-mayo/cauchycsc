# Python file to help using and managing datasets for training/testing

import numpy as np
import os 
import random 
import cv2
import pandas as pd
import mnist
import nibabel as nib
import torch
import torchvision

from sc import dictionary_learning as dl
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.decomposition import SparseCoder
from sklearn.preprocessing import normalize
from skimage.color import rgb2gray

from torchvision import transforms

from scipy.sparse import csr_matrix
from scipy.sparse import vstack

from helpers import img_utils as imu
from helpers import neuroimg as nim
from helpers import progress_track as pt

# Getting the data to use in format [train, test]
def get_data(data_path, ratio, classbal=True,dataset_size=1.):
	# data_path:	path of the directory containing the directories with the images for 
	#				each class of interest
	# ratio:		percentage of data to use for training
	# classbal:		class balance required?
	data = {}
	idxs = {}
	classes = os.listdir(data_path)
	class_size = np.zeros(len(classes), dtype=np.int16)
	train = []
	test = []
	class_paths = []
	train_labels = []
	test_labels = []
	for i in range(len(classes)):
		class_paths.append(os.path.join(data_path, classes[i]))
		images = os.listdir(class_paths[i])
		class_size[i] = int(len(images))
		data[i] = images
		idxs[i] = np.arange(class_size[i])
	if (dataset_size <=1.):
		portion = dataset_size
		max_size = np.sum(class_size)
	else:
		portion=1.
		max_size=dataset_size
	if classbal:
		max_size = int(max_size/len(classes))
		n_samples = min(class_size)
		n_samples = min([n_samples, max_size])
		n_samples = int(n_samples*portion)
		train_size = int(n_samples*ratio)
		test_size = int(n_samples - train_size)
		for i in range(len(classes)):
			training_aux = np.array(random.sample(range(0,class_size[i]),train_size), dtype=np.int16)
			testing_aux = np.array(list(set(idxs[i]) - set(training_aux)))
			aux = np.array(random.sample(range(0,testing_aux.size),test_size), dtype=np.int16)
			for j in range(training_aux.size):
				train.append([os.path.join(class_paths[i],data[i][j]), classes[i]])
				train_labels.append(classes[i])
			for j in range(aux.size):
				test.append([os.path.join(class_paths[i],data[i][j]), classes[i]])
				test_labels.append(classes[i])
		while (len(train) < train_size ):
			chosen_idx = random.randint(0,len(test)-1)
			chosen_item = test[chosen_idx]
			chosen_label = test_labels[chosen_idx]
			train.append(chosen_item)
			test.remove(chosen_item)
	else:
		for i in range(len(classes)):
			n_samples = min(class_size[i],max_size)
			n_samples = int(n_samples*portion)
			train_size = int(n_samples*ratio)
			test_size = int(n_samples - train_size)
			training_aux = np.array(random.sample(range(0,class_size[i]),train_size), dtype=np.int16)
			testing_aux = np.array(list(set(idxs[i]) - set(training_aux)))
			aux = np.array(random.sample(range(0,testing_aux.size),test_size), dtype=np.int16)
			for j in range(training_aux.size):
				train.append(data[i][j])
			for j in range(aux.size):
				test.append(data[i][testing_aux[aux[i]]])
	if ratio < 1. :
		return [pd.DataFrame(train).sample(frac=1).reset_index(drop=True), pd.DataFrame(test).sample(frac=1).reset_index(drop=True)]
	else:
		return pd.DataFrame(train).sample(frac=1).reset_index(drop=True)


def get_data_dir(data_path, classbal=True,dataset_size=1.,ratio=0.5):
	# data_path:	path of the directory containing the directories with the images for 
	#				each class of interest
	# ratio:		percentage of data to use for training
	# classbal:		class balance required?
	data = {}
	idxs = {}
	classes = os.listdir(data_path)
	class_size = np.zeros(len(classes), dtype=np.int16)
	train = []
	test = []
	class_paths = []
	train_labels = []
	test_labels = []
	for i in range(len(classes)):
		class_paths.append(os.path.join(data_path, classes[i]))
		images = os.listdir(class_paths[i])
		class_size[i] = int(len(images))
		data[i] = images
		idxs[i] = np.arange(class_size[i])
	if (dataset_size <=1.):
		portion = dataset_size
		max_size = np.sum(class_size)
	else:
		portion=1.
		max_size=dataset_size
	if classbal:
		max_size = int(max_size/len(classes))
		n_samples = min(class_size)
		n_samples = min([n_samples, max_size])
		n_samples = int(n_samples*portion)
		train_size = int(n_samples*ratio)
		test_size = int(n_samples - train_size)
		for i in range(len(classes)):
			training_aux = np.array(random.sample(range(0,class_size[i]),train_size), dtype=np.int16)
			testing_aux = np.array(list(set(idxs[i]) - set(training_aux)))
			aux = np.array(random.sample(range(0,testing_aux.size),test_size), dtype=np.int16)
			for j in range(training_aux.size):
				train.append([os.path.join(class_paths[i],data[i][j]), classes[i]])
				train_labels.append(classes[i])
			for j in range(aux.size):
				test.append([os.path.join(class_paths[i],data[i][j]), classes[i]])
				test_labels.append(classes[i])
		while (len(train) < train_size ):
			chosen_idx = random.randint(0,len(test)-1)
			chosen_item = test[chosen_idx]
			chosen_label = test_labels[chosen_idx]
			train.append(chosen_item)
			test.remove(chosen_item)
	else:
		for i in range(len(classes)):
			n_samples = min(class_size[i],max_size)
			n_samples = int(n_samples*portion)
			train_size = int(n_samples*ratio)
			test_size = int(n_samples - train_size)
			training_aux = np.array(random.sample(range(0,class_size[i]),train_size), dtype=np.int16)
			testing_aux = np.array(list(set(idxs[i]) - set(training_aux)))
			aux = np.array(random.sample(range(0,testing_aux.size),test_size), dtype=np.int16)
			for j in range(training_aux.size):
				train.append(data[i][j])
			for j in range(aux.size):
				test.append(data[i][testing_aux[aux[i]]])
	if ratio < 1. :
		return [pd.DataFrame(train).sample(frac=1).reset_index(drop=True), pd.DataFrame(test).sample(frac=1).reset_index(drop=True)]
	else:
		return pd.DataFrame(train).sample(frac=1).reset_index(drop=True)


# Function to pre-process the data
# src_dir:		array of strings, each one of them being the path of an image within the dataset
def preprocess_dataset(src, patch_size, stride=1, normalize=False):
	# Getting the paths for each image within it
	total_datasets = len(src)
	y = np.array([0])
	for image_path in src:
		img = imu.preprocess_img(image_path)
		img_patches = imu.get_patches(img, patch_size, stride)
		#print(img_patches.shape)
		#print(img_patches.dtype)
		if y.shape[0] == 1:
			y = img_patches
		else:
			y = np.vstack([y, img_patches])
	y = y.reshape(y.shape[0], -1)
	#print(y.dtype)
	if normalize:
		y_mean = np.mean(y, axis=1)[:,np.newaxis]
		y_std = np.std(y, axis=1)[:,np.newaxis]
		y -= y_mean
		y /= y_std
		return [y, y_mean, y_std]
	return y


def preprocess_nifti(mri_paths, patch_size, axis, stride=1, normalize=False, stride_slices=1, tot_slices=None, remove_zeros=False):
	# Getting the paths for each image within it
	total_mri = len(mri_paths)
	y = np.array([0])
	if tot_slices != None:
		slice_sizes = np.zeros(total_mri*tot_slices)
	else:
		slice_sizes = None
	#for i in range(2):
	curr_slice = 0
	for i in range(total_mri):
		#pname = mri_paths.loc[i].split('/')[-1].split('.')[0]
		pname = ''
		slices = nim.get_slices(mri_paths.loc[i][0], mri_paths.loc[i][1], axis, pname, stride=stride_slices, tot_slices=tot_slices)
		for single_slice in slices:
			img_patches = imu.get_patches(single_slice["slice"].toarray(), patch_size, stride)
			#print(img_patches.shape)
			img_patches = img_patches.reshape(img_patches.shape[0], -1)
			#print(img_patches.shape)
			if remove_zeros:
				img_patches = img_patches[~np.all(img_patches == 0, axis=1)]
			if y.shape[0] == 1:
				y = img_patches
			else:
				y = np.vstack([y, img_patches])
			slice_sizes[curr_slice] = int(img_patches.shape[0])
			curr_slice += 1
	
	#print(y.dtype)
	if normalize:
		y_mean = np.mean(y, axis=1)[:,np.newaxis]
		y_std = np.std(y, axis=1)[:,np.newaxis]
		y -= y_mean
		y /= y_std
		return [y, y_mean, y_std, slice_sizes]
	return [y, slice_sizes]

def preprocess_nifti_slices(mri_slices_data, patch_size=0, axis='', mri_shape='', stride=1, normalize=False, hf=False):
	# Getting the paths for each image within it
	total_slices = len(mri_slices_data)
	y = np.array([0])
	#for i in range(2):
	slice_sizes = np.zeros(total_slices)
	for i in range(total_slices):
		if(type(mri_shape)) != str:
			mri = nim.fill_nii(nim.read_nii(mri_slices_data[0][i]),mri_shape)
		else:
			mri = nim.read_nii(mri_slices_data[0][i])
		if type(axis) != str:
			single_slice = nim.get_slices(mri, '', axis, '',hf=hf)[mri_slices_data[1][i]]["slice"].toarray()
		else:
			single_slice = nim.get_slices(mri, '', mri_slices_data[3][i], '',hf=hf)[mri_slices_data[1][i]]["slice"].toarray()
		single_slice = single_slice/np.max(single_slice)
		if patch_size != 0:
			img_patches = imu.get_patches(single_slice, patch_size, stride)
		else:
			img_patches = np.array([single_slice])
		print(img_patches.shape)
		#print(img_patches.dtype)
		if len(y.shape) == 1:
			y = img_patches
		else:
			y = np.vstack([y, img_patches])
		slice_sizes[i] = int(img_patches.shape[0])
	print(y.shape)
	y = y.reshape(y.shape[0], -1)
	print(y.shape)
	#print(y.dtype)
	if normalize:
		y_mean = np.mean(y, axis=1)[:,np.newaxis]
		y_std = np.std(y, axis=1)[:,np.newaxis]
		y -= y_mean
		y /= y_std
		return [y, y_mean, y_std, slice_sizes]
	return [y, slice_sizes]

def get_random_slices(data_path, csv_class_dir, ratio, axis, classbal=True,dataset_size=1.,print_axis=False):
	# data_path:	path of the directory containing the directories with the images for 
	#				each class of interest
	# ratio:		percentage of data to use for training
	# classbal:		class balance required?
	data = {}
	idxs = {}
	classes = os.listdir(data_path)
	class_size = np.zeros(len(classes), dtype=np.int64)
	train = []
	test = []
	class_paths = []
	train_labels = []
	test_labels = []
	for i in range(len(classes)):
		class_paths.append(os.path.join(data_path, classes[i]))
		class_images = pd.read_csv(os.path.join(csv_class_dir, "%s.csv" % classes[i]))
		list_slices = []
		for j in range(len(class_images)):
			mri_slice_path = os.path.join(data_path, classes[i], class_images.loc[j]['mri_path'])
			for k in range(int(class_images.loc[j]['axis%s'% axis])):
				if print_axis:
					list_slices.append([mri_slice_path, k, classes[i], axis])
				else:
					list_slices.append([mri_slice_path, k, classes[i]])
		data[i] = list_slices
		class_size[i] = int(len(list_slices))
		idxs[i] = np.arange(class_size[i])
	if (dataset_size <=1.):
		portion = dataset_size
		max_size = np.sum(class_size)
	else:
		portion=1.
		max_size = dataset_size
	if classbal:
		max_size = int(max_size/len(classes))
		n_samples = min(class_size)
		n_samples = min([n_samples, max_size])
		n_samples = int(n_samples*portion)
		#print(n_samples)
		train_size = int(n_samples*ratio)
		test_size = int(n_samples - train_size)
		#print(train_size)
		#print(test_size)
		#print('n_samples= %d, train_size = %d, test_size=%d' % (n_samples, train_size, test_size))
		for i in range(len(classes)):
			training_aux = np.array(random.sample(range(0,class_size[i]),train_size), dtype=np.int16)
			testing_aux = np.array(list(set(idxs[i]) - set(training_aux)))
			aux = np.array(random.sample(range(0,testing_aux.size),test_size), dtype=np.int16)
			for j in range(training_aux.size):
				#print(j, end=' ')
				#print(data[i][j])
				train.append(data[i][training_aux[j]])
				#train_labels.append(classes[i])
			for j in range(aux.size):
				test.append(data[i][training_aux[j]])
				#test_labels.append(classes[i])
		while (len(train) < train_size ):
			chosen_idx = random.randint(0,len(test)-1)
			chosen_item = test[chosen_idx]
			chosen_label = test_labels[chosen_idx]

			train.append(chosen_item)
			#train_labels.append(chosen_label)

			test.remove(chosen_item)
			#test_labels.remove(chosen_label)
	else:
		for i in range(len(classes)):
			n_samples = min(class_size[i],max_size)
			n_samples = int(n_samples*portion)
			train_size = int(n_samples*ratio)
			test_size = int(n_samples - train_size)
			training_aux = np.array(random.sample(range(0,class_size[i]),train_size), dtype=np.int16)
			testing_aux = np.array(list(set(idxs[i]) - set(training_aux)))
			aux = np.array(random.sample(range(0,testing_aux.size),test_size), dtype=np.int16)
			for j in range(training_aux.size):
				train.append(data[i][j])
			for j in range(aux.size):
				test.append(data[i][testing_aux[aux[i]]])
	#return [train, train_labels, test, test_labels]
	if ratio < 1. :
		return [pd.DataFrame(train).sample(frac=1).reset_index(drop=True), pd.DataFrame(test).sample(frac=1).reset_index(drop=True)]
	else:
		return pd.DataFrame(train).sample(frac=1).reset_index(drop=True)

def get_dataset_coefficients(src_dir, dictionary, method, hist=True):
	img_coeff = np.zeros([len(src_dir), dictionary.shape[0]])
	patch_size = int(np.sqrt(dictionary.shape[1]))
	patch_size = (patch_size, patch_size)
	for i in range(len(src_dir)):
		face = imu.preprocess_img(src_dir[i],False)
		data = extract_patches_2d(face, patch_size)
		data = data.reshape(data.shape[0], -1)
		data_mean = np.mean(data, axis=1)[:, np.newaxis]
		data -= data_mean
		gamma = dl.transform(dictionary,data,method=method)
		if hist:
			gamma = np.sum(gamma,axis=0)
			img_coeff[i,:] = gamma.copy()
		else:
			img_coeff[i,:]  = csr_matrix((gamma.copy()).flatten())
	return img_coeff

def get_nifti_coefficients(mri_paths, mri_labels, dictionary, method, norm_shape, axis, hist=True):
	patch_size = int(np.sqrt(dictionary.shape[1]))
	patch_size = (patch_size, patch_size)
	#if hist:
	#	img_coeff = np.zeros([len(mri_paths), dictionary.shape[0]])
	#else:
	#	tot_feat = norm_shape.copy() - patch_size[0] + 1
	#	tot_feat[axis]=1
	#	tot_feat = np.prod(patch_size)*np.prod(tot_feat)
	#	img_coeff = np.zeros([len(mri_paths), tot_feat])
	scan=0
	labels = []
	for i in range(len(mri_paths)):
		pname = mri_paths[i].split('/')[-1].split('.')[0]
		mri = nim.fill_nii(nim.read_nii(mri_paths[i]),norm_shape)
		mri = mri/np.max(mri)
		slices = nim.get_slices(mri, mri_labels[i], axis, pname)
		for single_slice in slices:
			img_patches = extract_patches_2d(single_slice["slice"].toarray(), patch_size)
			#print(img_patches.shape)
			#print(img_patches.dtype)
			img_patches = img_patches.reshape(img_patches.shape[0], -1)
			img_patches_mean = np.mean(img_patches, axis=1)[:, np.newaxis]
			img_patches -= img_patches_mean
			gamma = dl.transform(dictionary,img_patches,method=method)
			#print(gamma.shape)
			if hist:
				gamma = np.sum(gamma,axis=0)
				#print(gamma.shape)
			else:
				gamma  = csr_matrix(gamma.toarray().flatten())
			if scan == 0:
				img_coeff = gamma.copy()
				scan +=1
			else:
				#print(img_coeff.shape)
				if hist:
					img_coeff = np.vstack([img_coeff, gamma])
				else:
					img_coeff = vstack([img_coeff, gamma])
			#scan += 1
			labels.append(single_slice["label"])
	return [img_coeff, labels]

# Python function that receives the patches to work with, the size of the slices involved in this, 
# the dictionary to obtain the coefficients and optionally the method to retrieve
# the sparse vectors. Returns an array containing all the coefficients.
def get_nifti_slice_coefficients(mri_patches, slice_sizes, dictionary, method="omp2", hist=True):
	mri_patches_mean = np.mean(mri_patches, axis=1)[:, np.newaxis]
	mri_patches -= mri_patches_mean
	if not hist:
		idxs = np.arange(mri_patches.shape[0])
		idxs = idxs[~np.all(mri_patches == 0, axis=1)]
		gamma = np.zeros([mri_patches.shape[0], dictionary.shape[0]])
	mri_patches = mri_patches[~np.all(mri_patches == 0, axis=1)]
	mri_patches = normalize(mri_patches, axis=1)
	#dictionary_norm = normalize(dictionary, axis=1)
	start_time = pt.get_time()
	coder = SparseCoder(dictionary=dictionary)
	if hist:
		gamma = coder.transform(mri_patches)
	else:
		gamma[idxs,:] = coder.transform(mri_patches)
	if (type(slice_sizes) != str):
		return slice_coefficients(gamma, slice_sizes, hist)
	else:
		return gamma

def slice_coefficients(gamma, slice_sizes, hist):
	start_idx = 0
	end_idx = int(slice_sizes[0])
	start_time = pt.get_time()
	print(hist)
	for i in range(len(slice_sizes)):
			if hist:
				gamma_aux = np.sum(gamma[start_idx:end_idx,:],axis=0)
			else:
				gamma_aux  = csr_matrix(gamma[start_idx:end_idx,:].flatten())
			if i == 0:
				img_coeff = gamma_aux.copy()
			else:
				if hist:
					img_coeff = np.vstack([img_coeff, gamma_aux])
				else:
					img_coeff = vstack([img_coeff, gamma_aux])
			start_idx = end_idx
			end_idx = int(np.sum(slice_sizes[0:i+2]))
	return img_coeff

def feature_coefficients(gamma, slice_sizes):
	start_idx = 0
	end_idx = int(slice_sizes[0])
	start_time = pt.get_time()
	print(hist)
	for i in range(len(slice_sizes)):
			if hist:
				gamma_aux = np.sum(gamma[start_idx:end_idx,:],axis=0)
			else:
				gamma_aux  = csr_matrix(gamma[start_idx:end_idx,:].flatten())
			if i == 0:
				img_coeff = gamma_aux.copy()
			else:
				if hist:
					img_coeff = np.vstack([img_coeff, gamma_aux])
				else:
					img_coeff = vstack([img_coeff, gamma_aux])
			start_idx = end_idx
			end_idx = int(np.sum(slice_sizes[0:i+2]))
	return img_coeff

def get_volume_coefficients(mri, dictionary, axis='', mri_shape='', hist= True, stride_patches=1, tot_slices=None, stride_slices=1, sliced=False):
	# Getting the paths for each image within it
	start_time = pt.get_time()
	if (type(mri) == str) or (type(mri) == np.ndarray):
		if (type(mri) == str):
			mri_vol = nim.read_nii(mri)
		else:
			mri_vol = mri.copy()
		mri_vol = nim.fill_nii(mri_vol, mri_shape)
		slices = nim.get_slices(mri_vol, '', axis, '', tot_slices=tot_slices, stride=stride_slices)
	elif (type(mri) == list):
		slices = mri

	patch_size = int(np.sqrt(dictionary.shape[1]))
	patch_size = (patch_size, patch_size)
	
	total_slices = len(slices)
	mri_patches = np.array([0])
	slice_sizes = np.zeros(total_slices)
	for i in range(total_slices):
		single_slice = slices[i]["slice"].toarray()

		single_slice = single_slice/np.max(single_slice)
		img_patches = imu.get_patches(single_slice, patch_size, stride_patches)
		if mri_patches.shape[0] == 1:
			mri_patches = img_patches
		else:
			mri_patches = np.vstack([mri_patches, img_patches])
		slice_sizes[i] = int(img_patches.shape[0])
	mri_patches = mri_patches.reshape(mri_patches.shape[0], -1)
	mri_patches_mean = np.mean(mri_patches, axis=1)[:, np.newaxis]
	mri_patches -= mri_patches_mean
	if sliced:
		print('here')
		img_coeff = get_nifti_slice_coefficients(mri_patches, slice_sizes, dictionary, hist=hist)
		return img_coeff
	else:
		img_coeff = get_nifti_slice_coefficients(mri_patches, '', dictionary, hist=hist)
		return [img_coeff, slice_sizes]

def get_data_labels(csv_path):
	file = open(csv_path, 'r')
	lines = file.readlines()
	file.close()
	data = []
	labels = []
	for line in lines:
		aux = line.replace("\n","").split(',')
		data.append(aux[0])
		labels.append(aux[1])
	return [data, labels]

def check_nii_dim(nii_dir):
	classes = os.listdir(nii_dir)
	class_size = np.zeros(len(classes), dtype=np.int16)
	class_paths = []
	data = {}
	for i in range(len(classes)):
		class_paths.append(os.path.join(nii_dir, classes[i]))
		images = os.listdir(class_paths[i])
		data[i] = images
	max_axis = np.array([0,0,0])
	for i in range(len(classes)):
		for j in range(len(data[i])):
			niishape = np.array(nim.read_nii(os.path.join(class_paths[i], data[i][j])).shape)
			if niishape[0] > max_axis[0]:
				max_axis[0] = niishape[0]
			if niishape[1] > max_axis[1]:
				max_axis[1] = niishape[1]
			if niishape[2] > max_axis[2]:
				max_axis[2] = niishape[2]
	return max_axis


def get_list_slices(nii_dir, csv_dir):
	classes = os.listdir(nii_dir)
	axis_size = np.zeros(3)
	for i in range(len(classes)):
		images = os.listdir(os.path.join(nii_dir, classes[i]))
		mri_data = pd.DataFrame(columns=['mri_path', 'axis0', 'axis1' ,'axis2'])
		for im in range(len(images)):
			mri_path = os.path.join(nii_dir, classes[i],images[im])
			for axis in range(3):
				[start_idx, end_idx] = nim.get_slice_range(mri_path,axis)
				axis_size[axis] = end_idx - start_idx
			mri_data.loc[im] = [images[im], axis_size[0], axis_size[1], axis_size[2]]
		class_path = os.path.join(csv_dir, "%s.csv" % (classes[i]))
		mri_data.to_csv(class_path, index=False)


def unpickle(file):
	import pickle
	with open(file,'rb') as fo:
		data = pickle.load(fo, encoding='bytes')
	return data

def get_cifar_images(get_labels=False):
	cifar_path = r'C:\Users\jazma\phd\data\cifar-10-batches-py\data_batch_'
	labels = []
	for i in range(1,6):
		data = unpickle(cifar_path + str(i))
		if i == 1:
			X = data[b'data']
		else:
			X = np.append(X, data[b'data'], axis=0)
		labels += data[b'labels']
	X = X.reshape(X.shape[0], 3, 32, 32).transpose(0,2,3,1)

	X_gray = np.zeros([X.shape[0], 1, 32, 32])
	for i in range(X.shape[0]):
		X_gray[i,0, :] = rgb2gray(X[i])
	if get_labels:
		return X_gray, labels
	return X_gray

def get_fmnist_images(get_labels=False, train=True):
	fmnist_path = r'C:\Users\jazma\phd\data'
	labels = []
	transform = transforms.Compose([transforms.ToTensor()])
	dataset = torchvision.datasets.FashionMNIST(root=fmnist_path, train=train, transform=transform, download=True)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=10000, shuffle=False)
	for images, datalabels in dataloader:
		data = images.data.squeeze()
		if len(labels) == 0:
			X = data
		else:
			X = np.append(X, data, axis=0)
		labels += datalabels.data.tolist()
	if get_labels:
		return X, labels
	return X


def get_nd_dataset(dataset_path, T0, zero_mean=False, labels=False, get_indxs=False):
	if dataset_path == 'mnist':
		get_indxs = False
		X = mnist.train_images()
		X = X/255.
		dataset_size = X.shape[0]
		if (T0 > dataset_size) or (T0==0):
			T0 = dataset_size
		#X = X[0:T0,:]
		sample = random.sample(range(dataset_size), T0)
		X = X[sample, :]
		if zero_mean:
			for i in range(T0):
				X[i] -= np.mean(X[i])
		if labels:
			X_labels = mnist.train_labels()
			X_labels = X_labels[sample]
	elif dataset_path == "cifar":
		get_indxs = False
		X, X_labels = get_cifar_images(True)
		X = X/X.max()
		dataset_size = X.shape[0]
		if (T0 > dataset_size) or (T0==0):
			T0 = dataset_size
		#X = X[0:T0,:]
		sample = random.sample(range(dataset_size), T0)
		X = X[sample, :]
		if zero_mean:
			for i in range(T0):
				X[i] -= np.mean(X[i])
		if labels:
			X_labels = np.array(X_labels)[sample]
	elif dataset_path == "fmnist":
		get_indxs = False
		X, X_labels = get_fmnist_images(True)
		X = X/X.max()
		dataset_size = X.shape[0]
		if (T0 > dataset_size) or (T0==0):
			T0 = dataset_size
		#X = X[0:T0,:]
		sample = random.sample(range(dataset_size), T0)
		X = X[sample, :]
		if zero_mean:
			for i in range(T0):
				X[i] -= np.mean(X[i])
		if labels:
			X_labels = np.array(X_labels)[sample]
	else:
		if os.path.isdir(dataset_path):
			data = [os.path.join(dataset_path, x)  for x in os.listdir(dataset_path)]
		else:
			paths_file = open(dataset_path, "r") 
			data = paths_file.readlines()
			paths_file.close()
		dataset_size = len(data)
		if (T0 > dataset_size) or (T0==0):
			T0 = dataset_size
			samples = range(T0)
		else:
			samples = random.sample(range(dataset_size), T0)
		data_samples = []
		# Identifying the type of data to work with -- MRI or images
		if data[0].strip().endswith('.nii') or data[0].strip().endswith('.nii.gz'): # MRIs
			#print(data[0])
			input_shape = nib.load(data[0].strip()).shape 
			X = np.zeros(shape=[T0] + list(input_shape))
			c = 0
			for i in samples:
				# X[i,:,:] = cv2.imread(os.path.join(dataset_path, data[i]), 0)
				#print(data[i])
				data_samples.append(data[i].strip())
				img = nib.load(data[i].strip()).get_data()
				img = img/np.max(img)
				if zero_mean:
					img -= np.mean(img)
				X[c,:] = img
				c += 1
		else:
			input_shape = cv2.imread(data[0].strip(), 0).shape 
			X = np.zeros(shape=[T0] + list(input_shape))
			c = 0
			for i in samples:
				# X[i,:,:] = cv2.imread(os.path.join(dataset_path, data[i]), 0)
				img = cv2.imread(data[i].strip(), 0)
				img = img/np.max(img)
				if zero_mean:
					img -= np.mean(img)
				X[c,:] = img
				c += 1
				data_samples.append(data[i].strip())
		if labels:
			X_labels = []
			for i in samples:
				X_labels.append( data[i].split(os.sep)[-2])
			X_labels = np.array(X_labels)

	if labels:
		if get_indxs:
			return X, X_labels, data_samples
		else:
			return X, X_labels
	else:
		if get_indxs:
			return X, data_samples
		else:
			return X

def crop_mris(data, labels, seed=-1, class_specific='no', portion=1., patch_size=[50,50,50]):
	if class_specific != 'no':
		sample = np.where(labels == class_specific)
		sample = list(sample[0])
	else:
		sample = list(range(len(labels)))
	#print(sample)
	if portion <= 1. :
		tot_samples = int(portion*len(sample))
	else:
		tot_samples = int(portion)
	train_data = np.zeros([tot_samples] + patch_size)
	original_shape = data[0].shape
	ul_corner = [0,0,0]
	if seed != -1:
		random.seed(seed)
	for i in range(tot_samples):
		for j in [0,1,2]:
			ul_corner[j] = random.randint(0, original_shape[j]-patch_size[j]-1)
		chosen = random.choice(sample)
		if tot_samples <= len(data):
			sample.remove(chosen)
		#print("%d, %s " % (chosen, ul_corner))
		train_data[i] = data[chosen][ul_corner[0]:ul_corner[0]+patch_size[0], ul_corner[1]:ul_corner[1]+patch_size[1], ul_corner[2]:ul_corner[2]+patch_size[2]].copy()
	return train_data

def get_ds_properties(dataset_path, T0):
	input_shape = []
	dataset_size = 0
	data_dim = 2
	if dataset_path == 'mnist':
		X = mnist.train_images()
	elif dataset_path == "cifar":
		X, X_labels = get_cifar_images(True)
	elif dataset_path == "fmnist":
		X, X_labels = get_fmnist_images(True)
	else:
		if os.path.isdir(dataset_path):
			data = [os.path.join(dataset_path, x)  for x in os.listdir(dataset_path)]
		else:
			paths_file = open(dataset_path, "r") 
			data = paths_file.readlines()
			paths_file.close()
		dataset_size = len(data)
		# Identifying the type of data to work with -- MRI or images
		if data[0].strip().endswith('.nii') or data[0].strip().endswith('.nii.gz'): # MRIs
			#print(data[0])
			input_shape = nib.load(data[0].strip()).shape 
			data_dim = 3
		else:
			input_shape = cv2.imread(data[0].strip(), 0).shape 
	if dataset_size == 0:
		dataset_size = X.shape[0]
		input_shape = list(X.shape)[1:]
	if (dataset_size > T0) and (T0 != 0):
		dataset_size = T0
	return dataset_size, np.array(input_shape), data_dim