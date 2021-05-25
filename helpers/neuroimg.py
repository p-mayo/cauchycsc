import os
import numpy as np
import nibabel as nib
import csv
import fnmatch 
import cv2

from scipy.sparse import csr_matrix
from sporco import util

from helpers import progress_track as pt

def mha2nii(source_path, output_path):
	old_ext = ".mha"
	new_ext = '.nii'
	for path, subdirs, files in os.walk(source_path):
		for name in files:
			if fnmatch.fnmatch(name, ("*%s" % old_ext)):
				#print os.path.join(path, name)
				#print os.path.join(outputpath,(os.path.splitext(name)[0] + newext))
				mha_file = os.path.join(path, name)
				nii_file = os.path.join(output_path, name)
				nii_file = nii_file.replace(old_ext, new_ext)
				print(mha_file)


def read_nii(nii_path):
	return nib.load(nii_path).get_data()

def get_patient_data(csv_path, nii_path):
	csv_file  = open(csv_path, "r") 
	patients_data = csv_file.readlines()
	csv_file.close()
	#print(patients_data)
	patients = []
	#i=0
	for patient in patients_data:
		#print(patient)
		scan_names = patient.replace("\n","").split(',')
		#print(scan_names)
		#scan_names.remove('\n')
		scans = {}
		scans["pname"] = scan_names[0]
		scan_names.remove(scan_names[0])
		scan_name = fnmatch.filter(scan_names, '*Flair*')[0]
		scans["Flair"] = read_nii(os.path.join(nii_path, scan_name + ".nii"))
		scan_names.remove(scan_name)
		scan_name = fnmatch.filter(scan_names, '*T1.*')[0]
		scans["T1"] = read_nii(os.path.join(nii_path, scan_name + ".nii"))
		scan_names.remove(scan_name)
		scan_name = fnmatch.filter(scan_names, '*T1c*')[0]
		scans["T1c"] = read_nii(os.path.join(nii_path, scan_name + ".nii"))
		scan_names.remove(scan_name)
		scan_name = fnmatch.filter(scan_names, '*T2*')[0]
		scans["T2"] = read_nii(os.path.join(nii_path, scan_name + ".nii"))
		scan_names.remove(scan_name)
		scans["GT"] = read_nii(os.path.join(nii_path, scan_names[0] + ".nii"))
		patients.append(scans)
		#i+=1
	return patients

# Python function to get the set of slices from a single patient
def get_slices(mri_img, nii_gt, axis,pname, stride=1, tot_slices=None, hf=False):
	# axis: 
	# 0 - X
	# 1 - Y
	# 2 - Z
	#slices = 0
	#gt_per_slice = []
	#condition = 0
	slices = []
	if (type(mri_img) == str):
		mri_img = read_nii(mri_img)
	[start_slice, end_slice] = get_slice_range(mri_img,axis)
	middle = np.floor((end_slice + start_slice)/2) 
	if tot_slices != None:
		tot_slices = tot_slices*stride
		if (middle - (tot_slices/2)) > start_slice:
			start_slice = np.floor(middle - (tot_slices/2))
			end_slice = start_slice + tot_slices
	for i in range(int(start_slice), int(end_slice), int(stride)):
		#print(i)
		mri_slice = get_slice_axis(mri_img, i, axis).astype(np.float64)
		if mri_slice.max() > 0:
			#slices += 1
			#aux = np.unique(nii_gt[:,:,i])
			#gt_per_slice.append(aux)
			#if aux.shape[0] > 1:
			#	condition += 1
			#	print(aux)
			if hf:
				npd = 16
				fltlmbd = 5
				sl, sh = util.tikhonov_filter(mri_slice, fltlmbd, npd)
				mri_slice = sh
			input_label = {}
			input_label["pname"] = "%s_%i" % (pname,i)
			input_label["slice"] = csr_matrix(mri_slice)
			if (type(nii_gt) != np.ndarray):
				input_label["label"] = nii_gt
			else:
				input_label["label"] = np.unique(get_slice_axis(nii_gt, i, axis))
			slices.append(input_label)
	return slices

def get_slice_axis(mri_img, slice_idx, axis):
	if axis == 0:
		return mri_img[slice_idx, :, :]
	elif axis == 1:
		return mri_img[:, slice_idx, :]
	else:
		return mri_img[:, :, slice_idx]

# Python function to get the slices from a list of patients
def get_all_slices(patients, modality, axis):
	tot_patients = np.size(patients)
	slices = get_slices(norm_mri(patients[0][modality]),patients[0]["GT"],axis,patients[0]["pname"])
	for i in range(tot_patients):
		slices = np.append(slices, get_slices(norm_mri(patients[i][modality]),patients[i]["GT"],axis,patients[i]["pname"]))
	return slices

def norm_mri(scan):
	max_val = np.max(scan)
	return (scan/max_val)*255

def get_csvdata(source_path, csv_path):
	patients = os.listdir(source_path)
	file = open(csv_path, 'w')
	for patient in patients:
		scans = os.listdir(os.path.join(source_path, patient))
		file.write("%s,"%patient)
		for scan in scans:
			file.write("%s,"%scan)
		file.write("\n")
	file.close()

def build_datasets(patients, outputpath, axis, binary=True):
	# axis=2
	# Creating folders for each modality and, then, for each class. Binary is the default.
	# Checking folders existance (called after the "key" from the slice)
	modalities = list(patients[0].keys())
	modalities.remove('pname')
	modalities.remove('GT')
	# 0 - Control
	# 1 - Abnormal
	if binary:
		classes = np.array(['0','1'])
	else:
		classes = np.unique(patients[0]["GT"])
	for item in modalities:
		mod_dir = os.path.join(outputpath, item)
		if os.path.isdir(mod_dir)==False:
			os.mkdir(mod_dir)
		classdir = []
		datasets = {}
		for i in range(classes.size):
			datasets[classes[i]] = []
			classdir.append(os.path.join(mod_dir, classes[i]))
			if os.path.isdir(classdir[i]) == False:
				os.mkdir(classdir[i])
		slices = get_all_slices(patients, item, axis)
		for single_slice in slices:
			slice_name = "%s%s"%(single_slice["pname"],'.pickle')
			if np.size(single_slice["label"]) == 1:
				slice_path = os.path.join(classdir[0],slice_name)
				#single_slice["label"] = np.array([0])
				datasets['0'].append(single_slice["slice"])
				pt.save_progress(single_slice["slice"],slice_path)
			else:
				if binary:
					slice_path = os.path.join(classdir[1],slice_name)
					#single_slice["label"] = np.array([1])
					datasets['1'].append(single_slice["slice"])
					pt.save_progress(single_slice["slice"],slice_path)
				else:
					for clss in single_slice["label"]:
						if clss != 0:
							slice_path = os.path.join(classdir[clss],slice_name)
							#single_slice["label"] = np.array([clss])
							datasets[clss].append(single_slice["slice"])
							pt.save_progress(single_slice["slice"],slice_path)
		#print(classdir)
		#for i in range(len(classdir)):
		#	pickle_dir = os.path.join(classdir[i], "dataset_class_%s.pickle" % (classes[i]))
		#	#print(pickle_dir)
		#	pt.save_progress(datasets[classes[i]],pickle_dir)

def save_slice(slice_path,data):
	if os.path.isfile(slice_path) == False:
		cv2.imwrite(slice_path,data)

def fill_nii(old_nii, new_shape, center=True):
	new_mri = np.zeros(new_shape)
	if center:
		start_coord = (new_shape/2).astype(int) - (np.array(old_nii.shape)/2).astype(int)
		end_coord = start_coord + np.array(old_nii.shape)
	else:
		start_coord = np.array([0,0,0])
		end_coord = np.array(old_nii.shape)
	# getting the center of the 
	new_mri[start_coord[0]:end_coord[0],start_coord[1]:end_coord[1],start_coord[2]:end_coord[2]] = old_nii
	return new_mri

def get_slice_range(mri, axis):
	if (type(mri) == str):
		mri_vol = read_nii(mri)
	else:
		mri_vol = mri
	start = -1
	end = -1
	current_slice=0
	threshold = 0
	while (start == -1):
		if np.count_nonzero(get_slice_axis(mri_vol, current_slice,axis)) > threshold:
			start=current_slice
		current_slice += 1
	while (end == -1):
		if np.count_nonzero(get_slice_axis(mri_vol, current_slice,axis)) == threshold:
			end = current_slice
		current_slice += 1
	return [start, end]


if __name__ == "__main__":
	#rootpath = '/space/pm15334/data/BRATS2015/BRATS2015_Training/HGG/'
	#outputpath = '/space/pm15334/data/BRATS2015/BRATS2015_Training/HGGNII/'
	typeLesion = 'HGG'
	rootpath = '/space/pm15334/data/BRATS2015/BRATS2015_Training/' + typeLesion + '/'
	outputpath = '/space/pm15334/data/BRATS2015/BRATS2015_Training/' + typeLesion + 'NII/'
