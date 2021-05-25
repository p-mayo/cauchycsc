# Python script to test the Cauchy Distribution and it's optimization
# using autograd.
import os 
import autograd
import autograd.numpy as np
import autograd.scipy.signal
import scipy.special as scis
import nibabel as nib

from scipy import linalg, optimize
from autograd import grad
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize

from helpers import progress_track as pt
from helpers import utils as ut
from helpers import img_utils as imu
from helpers import plotting as plot
from helpers.math_helpers import get_cubic_root
from stats import cauchy
from csc import thresholding as th
from csc.utils import estimate_y, get_residual, plot_multiple_singals, sparsity_metric, dim

convolve = autograd.scipy.signal.convolve

GEN_CSV = 'results.csv'

def mom(x, order, per_trial=False):
	if per_trial:
		print(np.sum(np.abs(x)**order, axis=0)/x.shape[0])
		return np.sum(np.abs(x)**order, axis=0)/x.shape[0]
	else:
		return np.sum(np.abs(x)**order)/np.prod(x.shape)

def estimate_gamma(y, alpha=1., per_trial=False, p=0.4):
	s = p + 1
	mom_p = mom(y, p, per_trial)
	
	flom = ((2**s)*scis.gamma(s/2.)*scis.gamma(-(s-1)/alpha))/(alpha*np.sqrt(np.pi)*scis.gamma(-(s-1)/2.))
	# print("mom_p = %s, s = %f, flom = %f, alpha/p=%f" % (str(mom_p), s, flom, (alpha/p)))
	gamma = (mom_p/flom)**(alpha/p)
	return gamma

def cauchy_penalty(z, gamma):
	#reg = np.log(1/(np.pi*gamma))
	#return -np.sum(np.log((gamma/(np.pi*(gamma**2 + z**2)))))
	return np.sum(np.log(gamma**2 + z**2))
	#return np.sum(np.log(1+z**2))

def laplacian_penalty(z):
	return lp_penalty(z, 1.)

def lp_penalty(z, p):
	return np.sum(np.abs(z)**p)

def log_penalty(z, delta):
	return np.sum(np.log(1 + (np.abs(z)/delta)))
	#return np.sum(np.log(delta + np.abs(z)))

def hard_penalty(z):
	return np.count_nonzero(z)

def sparsity_penalty(z, prior, gamma=None):
	if prior == 'laplace':
		penalty = laplacian_penalty(z)
	elif prior == 'cauchy':
		penalty = cauchy_penalty(z, gamma)
	elif prior == 'lp':
		penalty = lp_penalty(z, gamma)
	elif prior == 'log':
		penalty = log_penalty(z, gamma)
	elif prior == "hard":
		penalty = hard_penalty(z)
	#print(penalty)
	return penalty

def reconstruction_error(z, d, y):
	if dim(z) == dim(d) == dim(y):
		y_hat = d.dot(z)
	else:
		y_hat = estimate_y(z, d)
	least_squares = np.sum(get_residual(y, y_hat)**2) # + np.dot(betas.T, np.sum(d**2, axis=0)-1.)
	return least_squares

def dual_cost(d, z, y, gamma, lmbda=1.0, prior='laplace'):
	# K d \in R^M
	# K z \in R^Q
	# 1 y \in R^P= M + Q -1
	least_squares = reconstruction_error(z, d, y)
	penalty = sparsity_penalty(z, prior, gamma)
	return least_squares + lmbda*penalty

def cost_cauchy_z(z, d, y, gamma, lmbda=1.0, prior='laplace'):
	# K d \in R^M
	# K z \in R^Q
	# 1 y \in R^P= M + Q -1
	#dConvZ = estimate_y(z, d)
	#least_squares = np.sum(get_residual(y, dConvZ)**2)
	#penalty = sparsity_penalty(z, prior, gamma)
	#return least_squares + lmbda*penalty
	return dual_cost(d, z, y, gamma, lmbda, prior)

def cost_cauchy_d(d, z, y, gamma, lmbda=1.0, prior='laplace'):
	# K d \in R^M
	# K z \in R^Q
	# 1 y \in R^P= M + Q -1
	return dual_cost(d, z, y, gamma, lmbda, prior)

def cost_cauchy_b(betas, d, z, y, gamma, lmbda=1.0, prior='laplace'):
	# K d \in R^M
	# K z \in R^Q
	# 1 y \in R^P= M + Q -1
	#if type(delta) == type(None):
	#	delta = np.zeros(z.shape)
	#if type(betas) == type(None):
	#	betas = np.zeros((d.shape[1],1))
	#dConvZ = estimate_y(z, d)
	#least_squares = np.sum(get_residual(y, dConvZ)**2) + np.dot(betas.T, np.dot(d.T,d)-1.)
	#cauchy_penalty = np.sum(np.log(gamma/(np.pi*(gamma**2 + (z-delta)**2))))
	#return least_squares - lmbda*cauchy_penalty
	return dual_cost(d, z, y, gamma, betas, lmbda, prior)

def grad_cauchy_z(z, d, y, gamma, lmbda=1.0):
	# K d \in R^M
	# K z \in R^Q
	# 1 y \in R^P= M + Q -1
	M,K = d.shape
	Q,K = z.shape
	
	dConvZ = estimate_y(z, d)
	residual = (y - dConvZ)

	grad_z = np.zeros(z.shape)
	for k in range(K):
		corr = mh.build_toeplitz(-2*d[:,k],Q).T
		grad_z[:,k] = np.dot(corr, residual).reshape([Q])

	grad_z += lmbda*(2*z)/(gamma**2+z**2)
	#print("Grad Z shape: %s" % str(grad_z.shape))
	return grad_z

def grad_cauchy_d(d, z, y, gamma, lmbda=1.0):
	# K d \in R^M
	# K z \in R^Q
	# 1 y \in R^P= M + Q -1
	Q,K = z.shape
	if np.size(d.shape) != 2:
		d = d.reshape(int(d.shape[0])/K,K)
	M,K = d.shape
	dConvZ = estimate_y(z, d)
	residual = (y - dConvZ)
	#print("dConvZ.shape = %s" % str(dConvZ.shape))
	#print("residual.shape = %s" % str(residual.shape))
	grad_d = np.zeros(d.shape)
	for k in range(K):
		corr = mh.build_toeplitz(-2*z[:,k],M).T
		#print("Corr D shape: %s" % str(corr.shape))
		grad_d[:,k] = np.dot(corr, residual).reshape([M])
	#print("Grad D shape: %s" % str(grad_d.shape))
	return grad_d

def estimate_alpha(y):
	alpha = 1.
	return alpha

def learn_z(z, epsilon, p=10, alpha=0):
	return None

def irls(d, z, y, p, stopping):
	n = 0
	epsilon = 1.
	while epsilon > stopping:
		n += 1
		w = (z_old**2 + epsilon)**((p/2)-1)
		Q = w ** (-1)
		z_new = np.dot(Q, d.T).dot(np.inv())
		# if np.linalg.norm(z - z_new)**2 
	return z


def log_cost(z, d, y, gamma, prior, lmbda, log_path, min_sparse_err, stopping, i, new_cost, lr=None):
	#overall = cost_cauchy_z(z,d,y,gamma=gamma,lmbda=lmbda, prior=prior) - min_sparse_err
	sparsity = lmbda*sparsity_penalty(z, prior, gamma) - min_sparse_err
	rec_err = reconstruction_error(z, d, y)
	#overall = rec_err + sparsity - min_sparse_err
	overall = new_cost - min_sparse_err
	if lr != None:
		pt.log_event("%s [%04d %04d] COST = %f (!!!!!), Reconstruction error: %f, Sparsity penalty (%s prior): %f" 
					% (pt.get_time(), stopping, i, overall, rec_err, prior, sparsity), log_path)
		pt.log_event("%s        Reducing learning rate. New value: %f" 	% (pt.get_time(), lr), log_path)

	else:
		pt.log_event("%s [%04d %04d] COST = %f, Reconstruction error: %f, Sparsity penalty (%s prior): %f" 
					% (pt.get_time(), stopping, i, overall, rec_err, prior, sparsity), log_path)

# Possible priors:
# * Cauchy
# * Log
# * Laplace
def learn_d_z(initial_d, initial_z, y, stop, gamma=None, learning_rate=[0.02,0.0015],
				lmbda=1.0, prior='laplace', max_inner=25, path="", threshold=0., 
				sc_opt="thresholding", max_unchanged=0,save_filters=False, fixed=False,
				plot_hist=False, dataset="",dct_start=False, seed=0, get_results=False, 
				log_path="", bkp_path="", start_iter=1):
	# Initial values required for algorithm
	dict_shape = initial_d.shape
	initial_lr = learning_rate.copy()
	current_lr = initial_lr.copy()
	d = normalize(initial_d.reshape(dict_shape[0],-1),axis=1).reshape(dict_shape)
	z = initial_z.copy()
	#K, M = d.shape
	#T, K, Q = z.shape
	#P = M + Q - 1
	imgdim = dim(d)
	# Getting the gradient for the functions
	grad_cost_z = grad(cost_cauchy_z)
	grad_cost_d = grad(cost_cauchy_d)
	if sc_opt == 'ista':
		grad_sparsity = grad(sparsity_penalty)

	if max_unchanged == 0:
		max_unchanged = int(max_inner*.5)

	params = {} # Required when calling the iterative thresholding algorithm
	# params["lr"] = learning_rate[1] # This is not really needed...

	# To show increase in the same "origin"
	min_sparse_err = lmbda*sparsity_penalty(z, prior, gamma)

	# To keep record of the cost through the learning
	cost_hist = []
	rec_err = []
	sparse_err = []
	rec_err.append(reconstruction_error(z, d, y))
	sparse_err.append(min_sparse_err)
	cost_hist.append(cost_cauchy_z(z,d,y,gamma=gamma,lmbda=lmbda, prior=prior) - min_sparse_err)
	exp_time = pt.get_time_fileformat()	

	if type(max_inner) != list:
		max_inner = [max_inner, max_inner]

	# Getting paths ready for logs and results
	if path != "":
		if prior == 'cauchy':
			res_path = os.path.join(path, prior, sc_opt, "lambda_%s" % (str(lmbda)), "gamma_%s" % (str(gamma)), exp_time)
		elif prior == 'log':
			res_path = os.path.join(path, prior, sc_opt, "lambda_%s" % (str(lmbda)), "delta_%s" % (str(gamma)), exp_time)
		else:
			res_path = os.path.join(path, prior, sc_opt, "lambda_%s" % (str(lmbda)), exp_time)
		csv_path = os.path.join(res_path, "%s_%s.csv" % (exp_time, prior))
		log_path = os.path.join(res_path, "%s_%s_log.txt" % (exp_time, prior))
		#bkp_path = res_path
		#GEN_CSV = os.path.join(path, "results.csv")
		ut.create_path(path)
		ut.create_path(res_path)
		X_hat = estimate_y(z,d)
		results = {}
		results["outer_it"] = 0
		results["overall_cost"] = cost_hist[-1]
		results["rec_err"] = rec_err[-1]
		results["sparse_err"] = sparse_err[-1]
		results["sparsity_avg_fm"] = sparsity_metric(z, mode="fm")
		results["sparsity_l0"] = sparsity_metric(z, mode="l0")
		results["avg_psnr"] = imu.get_avg_psnr(y, X_hat)
		#atom_avg = sparsity_metric(z, mode="atom")
		#for i in range(atom_avg.shape[0]):
		#	results["sparsity_avg_atom_%d" % (i+1)] = atom_avg[i]
		pt.save_results(results, csv_path)
	else:
		csv_path = ""
		res_path = ""
	pt.log_event("%s ************************* STARTING ************************" % (pt.get_time()), log_path)
	pt.log_event("%s \tEXP ID:               %s " % (pt.get_time(), str(exp_time)), log_path)
	if dataset != "":
		pt.log_event("%s \tDataset:              %s " % (pt.get_time(), str(dataset)), log_path)
	pt.log_event("%s \tLambda:               %s " % (pt.get_time(), str(lmbda)), log_path)
	pt.log_event("%s \tPrior:                %s" % (pt.get_time(), prior.title()), log_path)
	pt.log_event("%s \tTrainsize:            %s" % (pt.get_time(), str(y.shape)), log_path)

	# ============================================== PARAMETERS FOR THE DIFFERENT PRIORS
	# ---------------------------------------------- SETTINGS FOR LOG THRESHOLDING
	# Learning rate and lmbda does not change for the method
	# Generally speaking... lmbda absorbs the learning rate!!!!
	# So I can change the value of lambda for the thresholding functions by this...
	params["lmbda"] = lmbda*learning_rate[1]
	if prior == "log":
		threshold = (np.sqrt(2*params["lmbda"]) - gamma)#*learning_rate[1]
		params["delta"] = gamma
		common_title = r"under %s prior, $\lambda$ = %s, \$delta$ = %s\n(EXP_ID: %s)" % (prior.title(), lmbda, gamma, exp_time)
		pt.log_event("%s \t   Delta:             %s" % (pt.get_time(), str(gamma)), log_path)
		pt.log_event("%s \t   Threshold:         %s" % (pt.get_time(), threshold), log_path)
	elif (prior == "laplace") or (prior == "hard"):
		params["portion"] = 1.
		common_title = r"under %s prior, $\lambda$ = %s \n(EXP_ID: %s)" % (prior.title(), lmbda, exp_time)
		pt.log_event("%s \t   Threshold:         %s" % (pt.get_time(), params["lmbda"] ), log_path)
	elif prior == "cauchy":
		params["gamma"] = gamma
		#params["hard"] = True
		params["hard"] = 0.#th.get_cauchy_th(gamma, lmbda)
		common_title = r"under %s prior, $\lambda$ = %s, \$gamma$ = %s\n(EXP_ID: %s)" % (prior.title(), lmbda, gamma, exp_time)
		pt.log_event("%s \t   Gamma:             %s " % (pt.get_time(), str(gamma)), log_path)
		pt.log_event("%s \t   Threshold:         %s " % (pt.get_time(), str(params["hard"])), log_path)

	pt.log_event("%s \tStopping criteria:    %s" % (pt.get_time(), str(stop)), log_path)
	pt.log_event("%s \tInner loop:           %s" % (pt.get_time(), str(max_inner)), log_path)
	pt.log_event("%s \tAlgorithm:            %s" % (pt.get_time(), sc_opt.upper()), log_path)	
	pt.log_event("%s \tLearning rate [d z]:  %s" % (pt.get_time(), str(learning_rate)), log_path)	
	pt.log_event("%s \tExperiment CSV:       %s" % (pt.get_time(), str(csv_path)), log_path)	
	pt.log_event("%s \tGeneral CSV:          %s" % (pt.get_time(), str(GEN_CSV)), log_path)	

	# Adapting stopping criteria
	if type(stop) == int:
		stopping = start_iter
	else:
		stopping = stop
		stop *= 10

	# Logging initial values
	pt.log_event("\n\n%s ************************* TRAINING START ************************\n" % (pt.get_time()), log_path)	
	#pt.log_event("%s [%04d %04d] COST = %f, Reconstruction error: %f, Sparsity penalty (%s prior): %f" 
	#	% (pt.get_time(), 0, 0, cost_hist[-1], rec_err[-1], prior, sparse_err[-1]), log_path)
	# log_cost(z, d, y, gamma, prior, lmbda, log_path, min_sparse_err, 0, 0, rec_err[-1] + sparse_err[-1])

	# Main training loops starts here
	if path != "":
		X_hat = estimate_y(z,d)
		# ------------------------- PLOTTING EVERYTHING IN THE SAME PLOT
		if imgdim == 1:
			img_path = os.path.join(res_path,"learning_iter_0.png")
			img_title = "Learning in iteration 0 %s" % (common_title)
			plot_multiple_singals([d, X_hat, z], T=[0,1,2,3,4,5], title=img_title, path = img_path)
		elif imgdim == 2:
			img_path = os.path.join(res_path,"learning_recons_iter_0.png")
			img_title = "Reconstruction in iteration 0 %s" % (common_title)
			imu.show_imgs(X_hat, img_title, img_path)
			img_path = os.path.join(res_path,"learning_filters_iter_0.png")
			img_title = "Filter learned in iteration 0 %s" % (common_title)
			imu.show_imgs(d, img_title, img_path)
	learning_start = pt.get_time()
	while stopping <= stop:
		# learning the coefficients
		z_old = z.copy()
		#old_cost = reconstruction_error(z, d, y)
		old_cost = cost_cauchy_z(z,d,y,gamma=gamma,lmbda=lmbda, prior=prior)
		pt.log_event("\n%s [---------] Learning the coefficients. Out It = %04d" % (pt.get_time(), stopping), log_path)
		log_cost(z, d, y, gamma, prior, lmbda, log_path, min_sparse_err, stopping, -1, old_cost)
		unchanged = 0
		#current_lr = initial_lr.copy()
		for i in range(max_inner[1]):
			if sc_opt == "gd":
				z = z - current_lr[1]*grad_cost_z(z, d, y, gamma=gamma, lmbda=lmbda, prior=prior)
			elif sc_opt == "thresholding":
				#print(current_lr[1])
				#print(d[0,:,:])
				z_aux = grad_cost_z(z.copy(), d, y, gamma=gamma, lmbda=0.0, prior=prior)
				#print(old_cost)
				#print(y[0,0])
				#print(z_aux[0,0,0])
				z = z - current_lr[1]*z_aux
				z = th.shrink(z, prior, params)
			# new_cost = reconstruction_error(z, d, y)
			new_cost = cost_cauchy_z(z,d,y,gamma=gamma,lmbda=lmbda, prior=prior)
			if (new_cost >= old_cost) and (not (prior in ["hard", "log"] and i == 0)):
				if new_cost == old_cost:
					unchanged += 1
					if unchanged >= max_unchanged:
						#current_lr[1] = initial_lr[1]
						pt.log_event("\n%s [---------] Coefficients are not changing. Stopping at inner iteration = %04d" 
										% (pt.get_time(), i), log_path)
						break
				current_lr[1] /= 2
				log_cost(z, d, y, gamma, prior, lmbda, log_path, min_sparse_err, stopping, i, new_cost, current_lr[1])
				params["lmbda"] = lmbda*current_lr[1]
				z = z_old.copy()
			else:
				unchanged = 0
				z_old = z.copy()
				old_cost = new_cost
				log_cost(z, d, y, gamma, prior, lmbda, log_path, min_sparse_err, stopping, i, new_cost)
			#return estimate_y(z,d)

		# learn basis
		if not fixed:
			d_old = d.copy()
			unchanged = 0
			#old_cost = reconstruction_error(z, d, y)
			old_cost = cost_cauchy_z(z,d,y,gamma=gamma,lmbda=lmbda, prior=prior)
			pt.log_event("\n%s [---------] Learning the dictionary. Out It = %04d" % (pt.get_time(), stopping), log_path)
			log_cost(z, d, y, gamma, prior, lmbda, log_path, min_sparse_err, stopping, -1, old_cost)
			for i in range(max_inner[0]):
				d = d - current_lr[0]*grad_cost_d(d, z, y, gamma, lmbda=lmbda, prior=prior)
				#d = normalize(d,axis=1)
				d = normalize(d.reshape(dict_shape[0],-1),axis=1).reshape(dict_shape)
				#new_cost = reconstruction_error(z, d, y)
				new_cost = cost_cauchy_z(z,d,y,gamma=gamma,lmbda=lmbda, prior=prior)
				if new_cost >= old_cost:
					if new_cost == old_cost:
						unchanged += 1
						if unchanged >= max_unchanged:
							#current_lr[0] = initial_lr[0]
							pt.log_event("\n%s [---------] Dictionary atoms are not changing. Stopping at inner iteration = %04d" 
											% (pt.get_time(), i), log_path)
							break
					current_lr[0] /= 2
					log_cost(z, d, y, gamma, prior, lmbda, log_path, min_sparse_err, stopping, i, new_cost, current_lr[0])
					d = d_old.copy()
				else:
					unchanged = 0
					z_old = z.copy()
					old_cost = new_cost
					log_cost(z, d, y, gamma, prior, lmbda, log_path, min_sparse_err, stopping, i, new_cost)
		rec_err.append(reconstruction_error(z, d, y))
		sparse_err.append(lmbda*sparsity_penalty(z, prior, gamma))
		cost_hist.append(cost_cauchy_z(z,d,y,gamma=gamma,lmbda=lmbda, prior=prior) - min_sparse_err)
		X_hat = estimate_y(z,d)
		if bkp_path != "":
			filters_file = os.path.join(bkp_path, "filters_iter_%d.pckl" % (stopping))
			pt.save_progress(d, filters_file)

			# ------------------------- PLOTTING EVERYTHING IN THE SAME PLOT
		if path !="":
			#coefficients_file = os.path.join(bkp_path, "coefficients_iter_%d.pckl" % (stopping))
			#pt.save_progress(z, coefficients_file)
			if imgdim == 1:
				img_path = os.path.join(res_path,"learning_iter_%s.png" % (str(stopping)))
				img_title = "Learning in iteration %s %s" % (str(stopping), common_title)
				plot_multiple_singals([d, X_hat, z], T=[0,1,2,3,4,5], title=img_title, path = img_path)
			elif imgdim == 2:
				img_path = os.path.join(res_path,"learning_recons_iter_%s.png" % (str(stopping)))
				img_title = "Reconstruction in iteration %s %s" % (str(stopping), common_title)
				imu.show_imgs(X_hat, img_title, img_path)
				if not fixed:
					img_path = os.path.join(res_path,"learning_filters_iter_%s.png" % (str(stopping)))
					img_title = "Filter learned in iteration %s %s" % (str(stopping), common_title)
					imu.show_imgs(d, img_title, img_path)
			#elif imgdim ==3:
				#for t in range(X_hat.shape[0]):
				#	img = nib.Nifti1Image(X_hat[t], np.eye(4))
				#	nib.save(img, os.path.join(res_path, 'recons_brain_%d_iter_%d.nii' % (t, stopping)))

			# ------------------------- SAVING RESULTS IN A CSV FILE FOR EVERY ITERATION
			results = {}
			results["outer_it"] = stopping
			results["overall_cost"] = cost_hist[-1]
			results["rec_err"] = rec_err[-1]
			results["sparse_err"] = sparse_err[-1]
			results["sparsity_avg_fm"] = sparsity_metric(z, mode="fm")
			results["sparsity_l0"] = sparsity_metric(z, mode="l0")
			results["avg_psnr"] = imu.get_avg_psnr(y, X_hat)
			#atom_avg = sparsity_metric(z, mode="atom")
			#for i in range(atom_avg.shape[0]):
			#	results["sparsity_avg_atom_%d" % (i+1)] = atom_avg[i]
			pt.save_results(results, csv_path)
		#pt.log_event("%s [%04d ----] COST = %f, Reconstruction error: %f, Sparsity penalty (%s prior): %f" 
		#	% (pt.get_time(), stopping, cost_hist[-1], rec_err[-1], prior, sparse_err[-1]), log_path)
		#log_cost(z, d, y, gamma, prior, lmbda, log_path, min_sparse_err, stopping, i, rec_err[-1] + sparse_err[-1])
		if type(stop) == int:
			stopping += 1
		else:
			stop=cost_hist[-1]
	end_time = pt.get_time(learning_start)
	if (GEN_CSV != ""):
		# ------------------------- SAVING SUMMARY RESULTS IN A COMMON CSV
		results = {}
		results["exp_id"] = ("%s_%s" % (exp_time, prior))
		results["mode"] = "convolutional"
		results["dataset"] = dataset
		results["base_path"] = path
		results["prior"] = prior
		results["fixed"] = fixed
		results["dct_start"] = dct_start
		results["filt_size"] = dict_shape[-1]
		results["num_filt"] = dict_shape[0]
		results["num_samples"] = y.shape[0]
		results["sc_opt"] = sc_opt
		results["max_outer"] = stop
		results["max_inner"] = str(max_inner).replace(',', '')
		results["lmbda"] = lmbda
		results["param"] = gamma
		results["in_lr_dict"] = learning_rate[0]
		results["in_lr_sprse"] = learning_rate[1]
		results["curr_lr_dict"] = current_lr[0]
		results["curr_lr_sprse"] = current_lr[1]
		results["overall_cost"] = cost_hist[-1]
		results["rec_err"] = rec_err[-1]
		if "X_hat" not in locals():
			X_hat = estimate_y(z,d)
		results["avg_psnr"] = imu.get_avg_psnr(y, X_hat)
		results["raw_sparse_err"] = sparse_err[-1]
		results["adj_sparse_err"] = sparse_err[-1] - min_sparse_err
		results["sparsity_avg_fm"] = sparsity_metric(z, mode="fm")
		results["sparsity_l0"] = sparsity_metric(z, mode="l0")
		results["train_time"] = end_time
		results["res_path"] = res_path
		results["seed"] = seed
		#atom_avg = sparsity_metric(z, mode="atom")
		#for i in range(atom_avg.shape[0]):
		#	results["sparsity_avg_atom_%d" % (i+1)] = atom_avg[i]
		pt.save_results(results, GEN_CSV)
	if save_filters and (path != ""):
		filters_file = "%s_%s_filters.pckl" % (res_path, exp_time)
		pt.save_progress(d, filters_file)
		pt.log_event("\n\n%s *** The filters have been saved in:\n\t%s\n" % (pt.get_time(), filters_file), log_path)
	pt.log_event("\n\n%s *** Training time: \t\t%s\n" % (pt.get_time(), end_time), log_path)
	pt.log_event("\n\n%s ************************* TRAINING END ************************\n" % (pt.get_time()), log_path)	

	if plot_hist and (path != ""):
		# Check the distribution of the learned coefficients
		# If the prior is cauchy, re-estimate gamma to check if it agrees or not 
		# Histograms per feature and for all the dataset
		est_gamma = cauchy.estimate_gamma(z.ravel())
		img_path = os.path.join(res_path,"histogram_coeff_all.png")
		img_title = "Histogram of coefficients\ngamma = %s " % (str(est_gamma))
		plot.plot_histogram(z.ravel(), title=img_title, save_path=img_path)

		# shape[0] -- Samples
		# shape[1] -- n_filters
		# shape[2:] -- coefficientes
		for k in range(z.shape[1]): #for each feature across all feature maps
			est_gamma = cauchy.estimate_gamma(z[:,k,:,].ravel())
			img_path = os.path.join(res_path,"histogram_coeff_k%d.png" % ((k+1)))
			img_title = r"Histogram of coefficients k=%d\n$\gamma$ = %s " % ((k+1), str(est_gamma))
			plot.plot_histogram(z[:,k,:,].ravel(), title=img_title, save_path=img_path)

	if path == "":
		if get_results:
			return d, z, [cost_hist, rec_err, sparse_err], results
		else:
			return d,z, [cost_hist, rec_err, sparse_err]
	else:
		if get_results: 
			return d, z, csv_path, results
		else:
			return d,z, csv_path

