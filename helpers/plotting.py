# Plotting
import numpy as np

from matplotlib import pyplot as plt

from helpers import progress_track as pt

# Histograms

def plot_histogram(x, title="", save_path="", show=True):
	fig, ax = plt.subplots(1, 1)
	pt.save_progress(x, r'C:\phd\coeff.pckl')
	#ax.hist(x.ravel(), density=True, align='mid', bins=np.linspace(x.min(), x.max(), 51))
	#print(x.ravel().shape())
	#print(x.min())
	#print(x.max())
	ax.hist(x.ravel(), density=True, align='mid', bins=np.linspace(-.2, .2, 101))
	if title != "":
		ax.set_title(title)
	if save_path != "":
		plt.savefig(save_path,  bbox_inches='tight')
		plt.close()
	elif show:
		plt.show()
		plt.close()
	return fig