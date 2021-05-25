# Python script for several functions that could help for data analysis
import os

from autograd import numpy as np
from matplotlib import pyplot as plt

def plot_cost(costs, labels, col, title="", path=""):
	fig, ax = plt.subplots(figsize=(15,10))
	ymax = 0
	ymin = 0
	for data in costs:
		aux = np.ceil(data[col].max()*1.1)
		if aux > ymax:
			ymax = aux
		aux = np.ceil(data[col].min()*1.1)
		if aux < ymin:
			ymin = aux
	for data in costs:
		data[col].plot(ylim=(ymin,ymax))
	ax.legend(labels)
	plt.title(title)
	if path != "":
		plt.savefig(path,  bbox_inches='tight')
	else:
		plt.show()

def plot_costs(costs, labels, cols, path="", figsize=(15,5), show=True):
	fig, ax = plt.subplots(1, len(cols), figsize=figsize, sharex=True)
	ymax = 0
	ymin = 0
	c = 0
	for col in cols.keys():
		for data in costs:
			aux = np.ceil(data[col].max()*1.1)
			if aux > ymax:
				ymax = aux
			aux = np.ceil(data[col].min()*1.1)
			if aux < ymin:
				ymin = aux
		for data in costs:
			plt.sca(ax[c])
			if "psnr" in col:
				data[col].plot(ylim=(0,50))
			else:
				data[col].plot(ylim=(ymin,ymax))
			#df.plot(subplots)
		ax[c].legend(labels)
		ax[c].set_title("%s" % (cols[col]))

		c += 1
	if path != "":
		plt.savefig(path,  bbox_inches='tight')
		plt.close()
	elif show:
		plt.show()
		plt.close()
	return fig

def create_path(path):
	if(not os.path.exists(path)):
		folders = os.path.split(path)
		create_path(folders[0])
		os.mkdir(path)

def check_path(path):
	if not os.path.exists(path):
		folders = os.path.split(path)
		if len(folders[1]) > 0:
			return check_path(folders[0])
		else:
			return False
	else:
		return True