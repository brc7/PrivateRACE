import numpy as np
import scipy.stats
import scipy.special
import scipy.integrate
import math


'''
Uses ideas from Hall's thesis


'''


class SpectralDP():
	def __init__(self, epsilon, M, data, function, sensitivity, debug = False):
		# data is n x d data matrix
		# function is a callable with the following form: 
		# function(x,data) -> scalar 
		# where x is a d-dimensional vector and data is a n x d data matrix
		# sensitivity is the sensitivity of function(x,data)
		self.epsilon = epsilon
		self.d = data.shape[1] # dimensions
		n = data.shape[0]
		self.M = M # 2*M for Fourier? 

		# lattice is just all the points in the unit grid
		# bravais lattice for the FFT evaluation
		lattice = [Mi/M for Mi in range(0,self.M)]
		lattice_m = [Mi for Mi in range(0,self.M)] # k values corresponding to lattice entries
		args = [lattice for i in range(self.d)]
		self.flatlattice = np.array(np.meshgrid(*args))
		self.flatlattice = np.reshape(self.flatlattice,(self.d,-1)).T
		args = [lattice_m for i in range(self.d)]
		self.flatlattice_indices = np.array(np.meshgrid(*args))
		self.flatlattice_indices = np.reshape(self.flatlattice_indices,(self.d,-1)).T
		if debug: 
			print("Lattice evaluation points:")
			print(self.flatlattice)
		if debug: 
			print("Lattice evaluation indices:")
			print(self.flatlattice_indices)

		# This is the signal we will take the FFT of
		# X = np.zeros((self.k+1)**self.d)
		X = np.zeros([self.M for _ in range(self.d)]) # (k,k,k, ...d... k) array of values to feed into fftn

		for indices,point in zip(self.flatlattice_indices,self.flatlattice): 
			print(indices,point)
			# indices = index within X
			# point = point value of X
			X[indices] = function(point,data)

		if debug: 
			print("Value of X:")
			print(X)

		self.coefficients = np.fft.fftn(X) / self.flatlattice.shape[0]

		lam = 2*self.M**self.d / (epsilon * n)
		# self.coefficients += np.random.laplace(loc = 0.0, scale = lam, size = self.coefficients.shape)
		# self.coefficients += 1j*np.random.laplace(loc = 0.0, scale = lam, size = self.coefficients.shape)


		if debug:
			print("Fourier coefficients:")
			print(self.coefficients.shape)

	def query(self,q): # q is the d-dimensional query in the unit hypercube in R^d
		
		output = 0.0
		for indices in self.flatlattice_indices:
			coeff = self.coefficients[indices]
			output += coeff.real*np.cos(-2*np.pi*indices*q)
			output += coeff.imag*np.sin(-2*np.pi*indices*q)

		return output
