import numpy as np 

class RACE():
	def __init__(self, repetitions, hash_range, dtype = np.int32):
		self.dtype = dtype
		self.R = repetitions # number of ACEs (rows) in the array
		self.W = hash_range  # range of each ACE (width of each row)
		self.counts = np.zeros((self.R,self.W),dtype = self.dtype)
		self.real_counts = np.zeros((self.R,self.W),dtype = self.dtype)

	def add(self, hashvalues): 
		for idx, hashvalue in enumerate(hashvalues): 
			rehash = int(hashvalue)
			rehash = rehash % self.W
			self.real_counts[idx,rehash] += 1

	def remove(self, hashvalues): 
		for idx, hashvalue in enumerate(hashvalues): 
			rehash = int(hashvalue)
			rehash = rehash % self.W
			self.real_counts[idx,rehash] += -1

	def set_epsilon(self, epsilon):
		# make the whole RACE sketch epsilon-differentially private
		N = np.sum(self.real_counts[0,:])
		noise = np.random.laplace(scale = self.R / (N * epsilon), size=self.real_counts.shape)
		noise = np.floor(noise)
		self.counts = self.real_counts + np.array(noise,dtype = self.dtype)

	def clear(self): 
		self.counts = np.zeros((self.R,self.W), dtype = self.dtype)

	def non_private_query(self, hashvalues):
		mean = 0
		N = np.sum(self.real_counts[0,:])
		for idx, hashvalue in enumerate(hashvalues): 
			rehash = int(hashvalue)
			rehash = rehash % self.W
			mean = mean + self.real_counts[idx,rehash]
		return mean/(self.R * N)

	def query(self, hashvalues):
		mean = 0
		N = np.sum(self.counts) / self.R
		for idx, hashvalue in enumerate(hashvalues): 
			rehash = int(hashvalue)
			rehash = rehash % self.W
			mean = mean + self.counts[idx,rehash]
		return mean/(self.R * N)

	def print(self):
		for i,row in enumerate(self.counts): 
			print(i,'| \t',end = '')
			for thing in row: 
				print(str(int(thing)).rjust(2),end = '|')
			print('\n',end = '')

	def counts(self): 
		return self.counts












