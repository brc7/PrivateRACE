import numpy as np 

class RACE():
	def __init__(self, repetitions, hash_range, dtype = np.int32):
		self.dtype = dtype
		self.R = repetitions # number of ACEs (rows) in the array
		self.W = hash_range  # range of each ACE (width of each row)
		self.counts = np.zeros((self.R,self.W),dtype = self.dtype)

	def add(self, hashvalues): 
		for idx, hashvalue in enumerate(hashvalues): 
			rehash = int(hashvalue)
			rehash = rehash % self.W
			self.counts[idx,rehash] += 1

	def remove(self, hashvalues): 
		for idx, hashvalue in enumerate(hashvalues): 
			rehash = int(hashvalue)
			rehash = rehash % self.W
			self.counts[idx,rehash] += -1

	def make_private(self, epsilon):
		# make the whole RACE sketch epsilon-differentially private
		N = np.sum(self.counts[0,:])
		noise = np.random.laplace(scale = self.R / (N * epsilon), size=self.counts.shape)
		noise = np.floor(noise)
		self.counts += np.array(noise,dtype = self.dtype)

	def clear(self): 
		self.counts = np.zeros((self.R,self.W), dtype = self.dtype)

	def query(self, hashvalues):
		mean = 0
		N = np.sum(self.counts[0,:])
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












