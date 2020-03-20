import numpy as np
from scipy.stats import norm # for P_L2
import math 
from scipy.special import ndtr

class L2LSH():
	def __init__(self, N, d, r): 
		# N = number of hashes 
		# d = dimensionality
		# r = "bandwidth"
		self.N = N
		self.d = d
		self.r = r

		# set up the gaussian random projection vectors
		self.W = np.random.normal(size = (N,d))
		self.b = np.random.uniform(low = 0,high = r,size = N)


	def hash(self,x): 
		return np.floor( (np.squeeze(np.dot(self.W,x)) + self.b)/self.r )


def P_L2(c,w):
	try: 
		p = 1 - 2*ndtr(-w/c) - 2.0/(np.sqrt(2*math.pi)*(w/c)) * (1 - np.exp(-0.5 * (w**2)/(c**2)))
		return p
	except:
		return 1
	# to fix nans, p[np.where(c == 0)] = 1

def P_SRP(x,y): 
	x_u = x / np.linalg.norm(x)
	y_u = y / np.linalg.norm(y)
	angle = np.arccos(np.clip(np.dot(x_u, y_u), -1.0, 1.0))
	return 1.0 - angle / np.pi




class SRPMulti():
	# multiple SRP hashes combined into a set of N hash codes
	def __init__(self, reps, d, p): 
		# reps = number of hashes (reps)
		# d = dimensionality
		# p = "bandwidth" = number of hashes (projections) per hash code
		self.N = reps*p
		self.d = d
		self.p = p

		# set up the gaussian random projection vectors
		self.W = np.random.normal(size = (self.N,d))
		self.powersOfTwo = np.array([2**i for i in range(self.N)])

	def hash(self,x): 
		# p is the number of concatenated hashes that go into each
		# of the final output hashes
		h = np.sign( np.dot(self.W,x) )
		h = np.clip( h, 0, 1)
		if self.p > 1:
			h = np.reshape(h,(-1,self.p))
			n_hashes = h.shape[0]
			powersOfTwo = np.array([2**i for i in range(self.p)])
			codes = np.zeros(n_hashes)
			for idx,hi in enumerate(h): 
				codes[idx] = np.dot(hi,powersOfTwo)
			return codes
		else: 
			return(h)





class SRP():
	def __init__(self, N, d): 
		# N = number of hashes 
		# d = dimensionality
		# r = "bandwidth"
		self.N = N
		self.d = d

		# set up the gaussian random projection vectors
		self.W = np.random.normal(size = (N,d))
		self.powersOfTwo = np.array([2**i for i in range(self.N)])

	def hash(self,x): 
		h = np.sign( np.dot(self.W,x) )
		h = np.clip( h, 0, 1)
		return np.dot( h, self.powersOfTwo)

	def hash_independent(self,x,p = 1): 
		# p is the number of concatenated hashes that go into each
		# of the final output hashes
		h = np.sign( np.dot(self.W,x) )
		h = np.clip( h, 0, 1)
		if p > 1:
			h = np.reshape(h,(-1,p))
			n_hashes = h.shape[0]
			powersOfTwo = np.array([2**i for i in range(p)])
			codes = np.zeros(n_hashes)
			for idx,hi in enumerate(h): 
				codes[idx] = np.dot(hi,powersOfTwo)
			return codes
		else: 
			return(h)


class SRP_ALSH():
	def __init__(self, n, p, m, d):
		# n = number of hashes
		# p = power of hashes
		# m = dimensionality of transformation
		# d = dimensionality of input/query
		self.n = n 
		self.p = p 
		self.d = d 
		self.m = m 

		# set up the gaussian random projection vectors
		self.W = np.random.normal(size = (self.n*self.p,self.d + self.m))
		self.powersOfTwo = np.array([2**i for i in range(self.p)])

	def hash_query(self, q): 
		q = np.pad(q,(0,self.m),'constant', constant_values = (0,0))
		return self._hash(q)

	def hash_input(self, x): 
		norm_x = np.linalg.norm(x)
		x = np.pad(x,(0,self.m),'constant', constant_values = (0,0))
		for i in range(self.m): 
			power = 2**(i+1)
			x[self.d + i] = 0.5 - norm_x**power
		return self._hash(x)

	def _hash(self, xt): 
		# xt = transformed input / query x
		h = np.sign( np.dot(self.W,xt) )
		h = np.clip( h, 0, 1)
		if self.p > 1:
			h = np.reshape(h,(-1,self.p))
			n_hashes = h.shape[0]
			codes = np.zeros(n_hashes)
			for idx,hi in enumerate(h): 
				codes[idx] = np.dot(hi,self.powersOfTwo)
			return codes
		else: 
			return (h)

class HPH(): # hyperplanehash
	def __init__(self, n, d):
		# n = number of hashes
		# p = power of hashes
		# m = dimensionality of transformation
		# d = dimensionality of input/query
		self.n = n
		self.p = 2
		self.d = d
		# self.m = m

		# set up the gaussian random projection vectors
		self.W_data = np.random.normal(size = (self.n*2,self.d))
		self.W_query = self.W_data.copy()
		self.W_query[::2,:] = -1*self.W_query[::2,:]
		self.powersOfTwo = np.array([2**i for i in range(2)])

	def hash_query(self, q): 
		h = np.sign( np.dot(self.W_query,q) )
		h = np.clip( h, 0, 1)
		h = np.reshape(h,(-1,2))
		n_hashes = h.shape[0]
		codes = np.zeros(n_hashes)
		for idx,hi in enumerate(h): 
			codes[idx] = np.dot(hi,self.powersOfTwo)
		return codes

	def hash_input(self, x): 
		h = np.sign( np.dot(self.W_data,x) )
		h = np.clip( h, 0, 1)
		h = np.reshape(h,(-1,2))
		n_hashes = h.shape[0]
		codes = np.zeros(n_hashes)
		for idx,hi in enumerate(h): 
			codes[idx] = np.dot(hi,self.powersOfTwo)
		return codes

