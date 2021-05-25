# Python code to manage miscelaneus functions for progress tracking.

import pickle 				# To save the variables once they have been processed
import datetime
import os 

from time import time

from helpers import general as gen

def __init__():
	datetime.datetime.now().strftime(get_str_format())

def save_progress(workspace, path):
	pickle.dump( workspace, open( path, "wb" ) )

def load_progress(path):
	workspace = pickle.load( open( path, "rb" ) )
	return workspace

def log_event(message, path=""):
	print(message)
	if path != "":
		with open(path, "a") as f:
			print("%s" % (message), file=f)

def write_csv(csv_path, data, mode='a'):
	file = open(csv_path, mode)
	if type(data) == str:
		data = [data]
	for row in data:
		file.write(row + "\n")
	file.close()

def save_results(results, res_dir):
	if os.path.isfile(res_dir) == False:
		write_csv(res_dir, gen.get_str(results.keys(),","))
	write_csv(res_dir,gen.get_str(results.values(),","))
	return None

def get_str_format():
	time_format = "%Y-%m-%d %H:%M"
	return time_format

def get_time(start_time=None):
	if start_time==None:
		return datetime.datetime.now()
	else:
		return datetime.datetime.now() - start_time
def get_time_fileformat():
	timestamp = datetime.datetime.now()
	return timestamp.strftime("%Y%m%d_%H%M%S")

def read_csv(csv_path):
	lines = read_file(csv_path)
	csv_data = {}
	csv_keys = lines[0].replace("\n","").split(',')
	for i in range(1,len(lines)):
		csv_line = lines[i].replace("\n","").split(',')
		csv_data_2 = {}
		for j in range(1,len(csv_line)):
			csv_data_2[csv_keys[j]] = csv_line[j]
		csv_data[csv_line[0]] = csv_data_2
	return csv_data

def read_csv2(csv_path):
	lines = read_file(csv_path)
	data = []
	for line in lines:
		data.append(line.replace("\n","").split(','))
	return data

def read_file(path):
	file  = open(path, "r")
	lines = file.readlines()
	file.close()
	return lines