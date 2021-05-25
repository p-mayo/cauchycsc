# Math Helpers
# 
import sys
import numpy as np
#import tables
import time

from scipy.linalg import toeplitz

# The cubic root of some number is not well done by numpy...
def get_cubic_root(x):
	if type(x) == float:
		x_cr = np.complex(np.sign(x)*(np.abs(x)**(1/3)))
	elif type(x) == complex:
		x_cr = x**(1/3)
	else:
		x_cr = x_cr = np.zeros(x.shape, dtype=np.complex)
		x_cr[x.imag == 0] = np.sign(x[x.imag == 0].real)*(np.abs(x[x.imag == 0].real)**(1/3))
		x_cr[x.imag != 0] = x[x.imag !=0 ]**(1/3)
	return x_cr

"""
def dot_prod(A, B):
	n_row = A.shape[0]
	n_col = B.shape[1]
	# using hdf5
	fileName_C = ('CArray_C%.4f.h5' % time.time())
	# Defines an atom of type float32:
	atom = tables.Float64Atom()
	shape = (n_row, n_col)
	Nchunk = 128  # ?
	chunkshape = (Nchunk, Nchunk)
	chunk_multiple = 1
	block_size = chunk_multiple * Nchunk

	# Create a PyTables writable file with name fileName_C
	# The in-memory representation of a PyTables file
	# http://www.pytables.org/usersguide/libref/file_class.html
	h5f_C = tables.open_file(fileName_C, 'w')
	res = None
	# Create a new chunked array
	# h5f_C.root = / (RootGroup) ''
	#              children := []
	#              The parent group from which the new array will hang.
	#              It can be a path string (for example ‘/level1/leaf5’), or a Group instance
	# 'CArray' =   The name of the new array
	# atom =       An Atom instance representing the type and shape of the atomic objects 
	#              to be saved.
	# shape =      The shape of the new array.
	# chunkshape=chunkshape = The shape of the data chunk to be read or written in a single
	#              HDF5 I/O operation. Filters are applied to those chunks of data. The
	#              dimensionality of chunkshape must be the same as that of shape. If None,
	#              a sensible value is calculated (which is recommended).
	try:
		C = h5f_C.create_carray(h5f_C.root, 'CArray', atom, shape, chunkshape=chunkshape)
		sz= block_size
		for i in range(0, A.shape[0], sz):
			for j in range(0, B.shape[1], sz):
				for k in range(0, A.shape[1], sz):
					C[i:i+sz,j:j+sz] += np.dot(A[i:i+sz,k:k+sz],B[k:k+sz,j:j+sz])
		res = np.array(C)
		h5f_C.close()
	except:
		#print(ErrorMessage)
		if h5f_C.isopen:
			h5f_C.close()
		print("Unexpected error: ", sys.exc_info()[0])
		print(sys.exc_info()[1])
	return res
"""

def _block_slices(dim_size, block_size):
    """Generator that yields slice objects for indexing into 
    sequential blocks of an array along a particular axis
    """
    count = 0
    while True:
        #print("%d, %d, %d" %((count, count + block_size, 1)))
        yield slice(count, count + block_size, 1)
        count += block_size
        if count > dim_size:
            raise StopIteration

def blockwise_dot(A, B, max_elements=int(2**27), out=None):
    """
    Computes the dot product of two matrices in a block-wise fashion. 
    Only blocks of `A` with a maximum size of `max_elements` will be 
    processed simultaneously.
    """

    m,  n = A.shape
    n1, o = B.shape

    if n1 != n:
        raise ValueError('matrices are not aligned')

    if A.flags.f_contiguous:
        # prioritize processing as many columns of A as possible
        max_cols = int(max(1, max_elements / m))
        max_rows = int(max_elements / max_cols)

    else:
        # prioritize processing as many rows of A as possible
        max_rows = int(max(1, max_elements / n))
        max_cols = int(max_elements / max_rows)

    if out is None:
        out = np.empty((m, o), dtype=np.result_type(A, B))
    elif out.shape != (m, o):
        raise ValueError('output array has incorrect dimensions')

    for mm in _block_slices(m, max_rows):
        #print(mm)
        out[mm, :] = 0
        for nn in _block_slices(n, max_cols):
            A_block = A[mm, nn].copy()  # copy to force a read
            out[mm, :] += np.dot(A_block, B[nn, :])
            del A_block

    return out

def pnorm(a, p, axis):
	norm = np.abs(a) ** p
	if axis == -1:
		norm = np.sum(norm)	
	else:
		norm = np.sum(norm, axis=axis)		
	return norm

"""
	V is the vector to build the Toeplitz Matrix from whilst N is the dimension
	of the vector this Toeplitz would be convolved with
"""
def build_toeplitz(V, N):
	M = V.shape[0]
	P = M + N - 1
	first_col = np.zeros([P,1])
	first_row = np.zeros([N,1])
	first_col[0:M,0] = V
	toep = toeplitz(first_col, first_row)
	return toep

# For Cardano's Formula
def get_p(a, b, c, d):
	p = (3*a*c-b**2)/(3*a)
	return p

def get_q(a, b, c, d):
	q = (2*(b**3)-9*a*b*c+27*(a**2)*d)/(27*(a**3))
	return q

def get_delta(p, q):
	return (q**2)/4 + (p**3)/27

def get_cauchy_threshold_coef(gamma):
	a = 4
	b = 12*gamma**2-1
	c = 12*gamma**4 - 20*gamma**2
	d = 4*gamma**6 + 8*gamma**4 + 4*gamma**2
	return a, b, c, d

def get_delta_gamma(gamma):
	a, b, c, d = get_cauchy_threshold_coef(gamma)
	p = get_p(a, b, c, d)
	q = get_q(a, b, c, d)
	return get_delta(p, q)
