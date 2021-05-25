# Python script containing general helpers

def get_str(list_trans, concatenator=" "):
	return concatenator.join(str(x).replace("\n","") for x in list_trans)


def str2bool(in_value):
	if in_value.lower() in 'false':
		return False
	return True