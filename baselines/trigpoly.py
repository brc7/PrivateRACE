import numpy as np
import scipy.stats
import scipy.special
import scipy.integrate
import math


'''
TrigPolyDP notes:
Implements "Differentially Private Data Releasing for Smooth Queries"
Uses trigonometric polynomials to interpolate smooth queries - smooth queries 
being functions evaluated over the dataset


I have reached out to the authors for C++ code (broken link)

this doesn't work rn 
I have no idea why 
MC integration is fine and I implemented exactly as detailed in paper 
but still not sure what's up here

:(


Bibliography: 
A Statistical Framework for Differential Privacy1
Hall's thesis: http://ra.adm.cs.cmu.edu/anon/usr0/ftp/home/anon/ml2011/CMU-ML-12-113.pdf
Differential Privacy for Functions and Functional Data

ESTABLISHING STATISTICAL PRIVACY FOR FUNCTIONAL DATA VIA FUNCTIONAL DENSITIES
Differentially Private Data Releasing for Smooth Queries with Synthetic Database Output

Apple: Local differential privacy
https://machinelearning.apple.com/docs/learning-with-privacy-at-scale/appledifferentialprivacysystem.pdf

DPCube: Differentially Private Histogram Release through Multidimensional Parti- tioning
On the Complexity of Differentially Private Data Release
The Composition Theorem for Differential Privacy
Differential Privacy via Wavelet Transforms
Differentially Private Data Release for Data Mining
A Simple and Practical Algorithm for Differentially Private Data Release
https://simons.berkeley.edu/sites/default/files/docs/1123/nikolovslides.pdf
https://www.pewresearch.org/internet/wp-content/uploads/sites/9/2019/11/Pew-Research-Center_PI_2019.11.15_Privacy_FINAL.pdf



Sketching is pretty straightforward. We just project the dataset into a basis of cosines 
To query, we have to get the coefficients associated with the function (smooth) query


'''

class TrigPolyDP():
	def __init__(self, epsilon, data, K = 1, debug = False):
		# data is n x d data matrix
		# epsilon is a value
		self.epsilon = epsilon
		self.d = data.shape[1] # dimensions
		self.n = data.shape[0] # dimensions
		self.K = K
		self.t = math.floor(self.n**(1.0 / (2*self.d + self.K)))
		print("AA: ", self.n**(1.0 / (2.0*self.d + self.K)),self.t)
		self.r = math.ceil(0.5*K + 1.5) # for H_{t,r}(x)

		self.a = np.array(self._get_jackson_a())
		if debug:
			print("H_{t,r} = H_{"+str(self.t)+","+str(self.r)+"}: coefficients")
			print(self.a)


		# lattice is just all the combinations of
		# [int,int,int,int,... d] where each int
		# can range from 0 to k. In the paper,
		# k = n**(1/(2*d + smoothness thing))
		# For the initiated, this is a Bravais lattice with the unit vector
		lattice = [ti for ti in range(0,self.t)]
		args = [lattice for i in range(self.d)]
		self.fpoints = np.array(np.meshgrid(*args)) # frequency locations for the cosine vector
		self.fpoints = np.reshape(self.fpoints,(self.d,-1)).T
		# fpoints is a (t**d, d) shaped matrix
		# in the paper, fpoints[idx,:] = vector m in Algorithm 1: Outputting the summary 

		if debug:
			print("Frequency points:")
			print(self.fpoints)

		# create interpolation values
		arccosValues = np.arccos(data)
		# intermediateValues = n x d x t matrix
		# at index [n,d,t]: contains cos(lattice[t]*arccos(data[n,d]))
		intermediateValues = np.multiply.outer(arccosValues,lattice)
		intermediateValues = np.cos(intermediateValues)
		self.interpolationValues = np.zeros(self.t**self.d)
		for ti,m in enumerate(self.fpoints):
			# s = np.zeros(self.n)
			prod = np.ones(self.n)
			for dim,mi in enumerate(m):
				prod = prod * intermediateValues[:,dim,mi]
				# s += intermediateValues[:,dim,mi]
			# self.interpolationValues[ti] = 1.0 / self.n * np.sum(prod)
			self.interpolationValues[ti] = 1.0 / self.n * np.sum(prod)

		if debug: 
			print("Interpolation Values:")
			print(self.interpolationValues)
			# for each vector m,
			# cos(m_d * theta_{n,d}) = intermediateValues[n,d,m_d]

	def _H(self,x): # H_{t,r}(s) in the paper
		tp = math.floor(self.t / self.r) + 1
		gamma = 0 # normalization factor in generalized jackson kernel
		for p in range(0,int(self.r - self.r / tp)): 
			gamma = gamma + (-1)**p * ( scipy.special.binom(2*self.r, p) * scipy.special.binom(self.r*(tp+1) - tp*p - 1,self.r*(tp-1) - tp*p) )
		gamma = gamma*2
		gamma = gamma*np.pi
		if gamma == 0: 
			gamma = 1.0 
		out = (np.sin(tp*x / 2.0 ) / np.sin(x / 2.0) )**(2**self.r)
		out *= 1.0 / gamma 
		return out

	def _get_jackson_a(self):
		N = self.t
		Fs = 2*N
		x = np.linspace(0, 2*np.pi, Fs + 2, endpoint=False)
		fx = self._H(x)
		fx[np.isnan(fx)] = 1
		y = np.fft.rfft(fx) / x.size
		y[1:] *= 2
		return y.real

	def _integral_arg(self, theta, l, k, function):
		# argument to the integral for monte carlo
		# should be integrated from -pi to pi on all dimensions
		# (hypercube around origin in Rd with side length 2pi)
		# l and k are d-dimensional vectors that decide which integral to do 
		# theta is the evaluation point of the integrand
		# print(function(np.cos(theta)))
		value = np.prod( np.cos( (l/k)*theta ) ) * function(np.cos(theta))
		return value

	def _get_m(self, l, k, function):
		# l and k are vectors of integers
		# such that li / ki = ni

		# monte carlo g(x) is (2*pi)**d
		NMC = 1000
		pts = np.random.uniform(low = -np.pi, high = np.pi, size = (NMC,self.d))
		integral = 0
		for pt in pts:
			integral += (1.0 / NMC) * self._integral_arg(pt, l, k, function)
		integral /= (2*np.pi)**self.d
		print("MC Integral: ",integral)

		# integal = scipy.integrate.quad(lambda x: self._integral_arg(pt, l, k, function), -np.pi, np.pi)
		# print("Real Integral: ",integral)
		print(scipy.special.binom(self.K+1,k))
		out = self.a[l] * ((-1)**k) * scipy.special.binom(self.K+1,k)*integral
		print("Individual, Product: ",out,np.prod(out))
		out = np.prod(out)

		return out

	def _get_coefficients(self,function):
		# this gets the coefficients
		coefficients = np.zeros_like(self.interpolationValues)
		klattice = [ki for ki in range(1,self.K + 2)]
		args = [klattice for i in range(self.d)]
		klattice = np.array(np.meshgrid(*args))
		klattice = np.reshape(klattice,(self.d,-1)).T

		print(klattice)

		for idx,ns in enumerate(self.fpoints):
			# li = ki * ni
			for ks in klattice:
				ls = ks*ns
				if np.max(ls) <= self.t:
					print("K,L,N:",ks,ls,ns)
					coefficients[idx] += self._get_m(ls,ks,function)
		coefficients *= (-1)**self.d
		print(coefficients)
		return coefficients

	def query(self,function):
		# q is the d-dimensional query in [-1,1] in R^d
		# Every time we query we need to do Monte Carlo or sparse grid integration
		c = self._get_coefficients(function)
		return np.dot(c,self.interpolationValues)


