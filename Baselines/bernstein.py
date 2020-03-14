import numpy as np
import scipy.stats

# Bernstein notes:
# should suffer from curse of dimensionality due to 
# the lattice - number of points in lattice (and hence privacy budget) blows up
# exponentially in high dimensions

# To do this with KDE queries and data that are outside the range (0,1)
# first normalize the data to be inside the range (0,1) using the affine transformation 
# dmax,dmin = data.max(), data.min()
# data = data - dmin
# data /= (dmax - dmin)
# 
# And then, normalize queries to be inside the range (0,1) using the same transformation 
# query = query - dmin 
# query /= (dmax - dmin)
# 
# and finally, if there is a KDE bandwidth parameter 
# for the L2 distance, multiply the sigma by 1 / (dmax - dmin)
# for the angular distance, no multiplication is necessary 
# for other distances ???? (who knows???)
'''
# normalize data for Bernstein polynomial approximation
self.mind = data.min()
self.maxd = data.max()

This is the transformation
data -= mind # set min to zero
if (mind != maxd):
	data /= (maxd - mind)
# now data is all between 0 and 1 and we can do the approximation
'''



class BernsteinDP():
	def __init__(self, epsilon, k, data, function, sensitivity,debug = False):
		# data is n x d data matrix
		# function is a callable with the following form: 
		# function(x,data) -> scalar 
		# where x is a d-dimensional vector and data is a n x d data matrix
		# sensitivity is the sensitivity of function(x,data)
		# Im pretty sensitive rn tbh

		self.epsilon = epsilon
		self.d = data.shape[1] # dimensions
		self.k = k
		# lattice is just all the points in the grid
		lattice = [ki/k for ki in range(0,self.k+1)]
		lattice_k = [ki for ki in range(0,self.k+1)] # k values corresponding to lattice entries
		# lattice = np.array(lattice)
		# pls dont ask what this does. It hammers the meshgrid output into 
		# a set of all the different d-dimensional selections of the lattice values
		# I wrote it but I don't know how it does. I just know what it does. 
		args = [lattice for i in range(self.d)]
		self.interpolationPoints = np.array(np.meshgrid(*args))
		self.interpolationPoints = np.reshape(self.interpolationPoints,(self.d,-1)).T
		if debug: 
			print("Interpolation points:")
			print(self.interpolationPoints)

		args = [lattice_k for i in range(self.d)]
		# This contains the values of K associated with each of the dimensions of each interpolation point
		# it's largely for convenience and bc I can't figure out how the heck meshgrid shuffles stuff around
		# and honestly it's 2am at this point and I can't be asked
		self.interpolationPointsK = np.array(np.meshgrid(*args)) # used to compute binomial pmf values
		self.interpolationPointsK = np.reshape(self.interpolationPointsK,(self.d,-1)).T
		if debug: 
			print("Interpolation values of K:")
			print(self.interpolationPointsK)

		self.interpolationValues = np.zeros((self.k+1)**self.d) # function eval'd at each interpolationPoint

		lam = sensitivity * (self.k+1)**self.d * 1.0 / epsilon # for laplace mechanism
		for idx, row in enumerate(self.interpolationPoints):
			self.interpolationValues[idx] = function(row,data)

		self.interpolationValues += np.random.laplace(loc = 0.0, scale = lam, size = (self.k+1)**self.d);
		if debug: 
			print("Interpolation values:")
			print(self.interpolationValues)

	def query(self,q): # q is the d-dimensional query in the unit hypercube in R^d
		# the bernstein basis polynomial is 
		# b_{v,n}(x) = binomial coefficient (n choose v) * x**v * (1 - x)**(n - v)
		# b_{v,n}(x) = scipy.stats.binom.pmf(n,v,x,loc = 0.0)
		output = 0.0
		for value,k_values in zip(self.interpolationValues,self.interpolationPointsK): 
			# compute binomial coefficient for this value of k
			coeff = 1.0
			for qi,v in zip(q,k_values):
				coeff *= scipy.stats.binom.pmf(v,self.k,qi,loc=0)
			output += coeff * value
		# somethign something "let's have a conference deadline every week, it'll be fun!"
		# *inaudible screaming*
		# *hideous screeching*
		# booming voice declares "You have reached the ninth circle of GRAD SCHOOL"
		# I'd like to thank the authors of this paper tho
		# The sufficient documentation has brought some semblence of happiness to my weary eyes
		return output

# HAHA YEAH IT WORKS 
# NOW I CAN SLEEP 






class ScalableBernsteinDP():
	def __init__(self, epsilon, k, data, function, sensitivity,debug = False):
		# Scalable version of the other one
		# Turns out that you get out of memory errors due to exponential blowup
		# lol oops
		# data is n x d data matrix
		# function is a callable with the following form
		# function(x,data) -> scalar
		# where x is a d-dimensional vector and data is a n x d data matrix
		# sensitivity is the sensitivity of function(x,data)
		self.epsilon = epsilon
		self.sensitivity = sensitivity
		self.d = data.shape[1] # dimensions
		self.k = k

		self.interpolationValuesGT = np.zeros((self.k+1)**self.d) # function eval'd at each interpolationPoint
		for idx, point in enumerate(self.lattice(self.k,self.d)):
			self.interpolationValuesGT[idx] = function(point*1.0/self.k,data)

		lam = self.sensitivity * (self.k+1)**self.d * 1.0 / epsilon # for laplace mechanism
		self.interpolationValues = self.interpolationValuesGT + np.random.laplace(loc = 0.0, scale = lam, size = (self.k+1)**self.d);
		# if debug: 
		# 	print("Interpolation values:")
		# 	print(self.interpolationValues)

	def set_epsilon(self, epsilon): 
		lam = self.sensitivity * (self.k+1)**self.d * 1.0 / epsilon # for laplace mechanism
		self.interpolationValues = self.interpolationValuesGT + np.random.laplace(loc = 0.0, scale = lam, size = (self.k+1)**self.d);

	def query(self,q): # q is the d-dimensional query in the unit hypercube in R^d
		# the bernstein basis polynomial is 
		# b_{v,n}(x) = binomial coefficient (n choose v) * x**v * (1 - x)**(n - v)
		# b_{v,n}(x) = scipy.stats.binom.pmf(n,v,x,loc = 0.0)
		output = 0.0
		for value,k_values in zip(self.interpolationValues,self.lattice(self.k,self.d)): 
			# compute binomial coefficient for this value of k
			coeff = 1.0
			for qi,v in zip(q,k_values):
				# print(v,self.k,qi)
				coeff *= scipy.stats.binom.pmf(v,self.k,qi,loc=0)
			output += coeff * value
		return output

	def lattice(self,k,d):
		# Generator, yields interpolation values
		vector = np.zeros(d)
		yield vector
		while np.sum(vector) < d*k:
			for i in reversed(range(d)):
				if vector[i] < k:
					vector[i] += 1
					break
				else:
					if i is not 0:
						vector[i] = 0
			yield vector




