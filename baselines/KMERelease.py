import numpy as np
import scipy.stats
from bisect import bisect
import sys 


'''
Implementation of KME NIPS paper - heavily inspired by the code
posted online at https://github.com/matejbalog/RKHS-private-database


another baseline: 
ESTABLISHING STATISTICAL PRIVACY FOR FUNCTIONAL DATA VIA FUNCTIONAL DENSITIES
ICML 


'''

class KMEReleaseDP_1():
	# Algorithm 1 from the paper
	def __init__(self, epsilon, M, data, function, sigma_public = 500, debug = False):
		self.debug = debug
		self.epsilon = epsilon
		self.d = data.shape[1] # dimensions
		self.N_private = data.shape[0] # N
		# self.maxM = M
		self.M = M

		# 1. Generate synthetic points
		# sigma_public = 500.0 # default from paper
		X_private = data
		self.X_public = np.random.normal(loc=np.zeros(self.d), scale=sigma_public, size=(self.M, self.d))

		# 2. Create kernel matrix means
		# Not actually necessary
		# self.K_xx = self._kernel_matrix(X_private, function)
		# self.K_xx_mean = np.mean(self.K_xx)

		# 3. Get cross kernel means
		self.K_zx_rmeans = self._cross_kernel_matrix_rmeans(self.X_public, X_private, function)

		# 4. Get public kernel matrix
		self.K_zz = self._kernel_matrix(self.X_public, function)

		# 5. Find the basis (B) and weights (alpha) for each data point
		self.B, self.alpha, self.parents = self._reweight_public_compute_alpha(self.K_zx_rmeans, self.K_zz)

		# set_m(): 
		num_dimensions = bisect(self.parents, self.M)
		self.alpha_part = self.alpha[:num_dimensions]
		self.B_part = self.B[:num_dimensions, :self.M]
		self.K_zz_part = self.K_zz[:self.M, :self.M]
		self.K_zx_rmeans_part = self.K_zx_rmeans[:self.M]

		# set_epsilon(): 
		sensitivity = 2.0 / np.sqrt(self.N_private)
		self.beta = self.alpha + np.random.laplace(scale = sensitivity / epsilon, size=len(self.alpha))

		# Now, the final output is self.X_public, self.beta
		# Each point in self.X_public is weighted by a value self.beta

	def set_m(self, m): 
		num_dimensions = bisect(self.parents, self.M)

		self.alpha_part = self.alpha[:num_dimensions]
		self.B_part = self.B[:num_dimensions, :self.M]
		self.K_zz_part = self.K_zz[:self.M, :self.M]
		self.K_zx_rmeans_part = self.K_zx_rmeans[:self.M]
		self.set_epsilon(self.epsilon)

	def set_epsilon(self, epsilon): 
		# It is really expensive to re-compute the kernel matrices all the time
		# so we can just reset epsilon and M without having to recompute
		# using set_epsilon
		# Then, all queries after calling this function will be at this level of epsilon
		# at least until you reset epsilon again
		self.epsilon = epsilon
		sensitivity = 2.0 / np.sqrt(self.N_private)
		self.beta = self.alpha + np.random.laplace(scale = sensitivity / self.epsilon, size=len(self.alpha))

	def _cross_kernel_matrix(self, X, Y, function):
		# X is (n x d) and Y is (m x d)
		# Returns (n x m) output Z where 
		# Z[i,j] = function(x[i,:],y[j,:])
		K = np.zeros((X.shape[0],Y.shape[0]))
		for i in range(X.shape[0]): 
			for j in range(Y.shape[0]): 
				K[i,j] = function(X[i,:],Y[j,:])
		return K

	def _cross_kernel_matrix_rmeans(self, X, Y, function): 

		M,N = len(X), len(Y)
		compute_threshold = 10000 ** 2
		if N*M <= compute_threshold: 
			K_zx = self._cross_kernel_matrix(X, Y, function)
			return np.mean(K_zx, axis = 1)
		else: 
			sums = np.zeros(M)
			step = compute_threshold // M 
			start = 0 
			while start < N: 
				sums += np.sum(self._cross_kernel_matrix(X, Y[start:(start+step)], function), axis=1)
				start = min(start + step, N)
				if self.debug: 
					print("Kernel matrix row means: ",100.0 * start / N)
					sys.stdout.flush()
			return sums / N 
		# self.K_zx = self._cross_kernel_matrix(self.X_public, X_private, function)
		# self.K_zx_rmeans = np.mean(self.K_zx, axis = 1)


	def _kernel_matrix(self, X, function):
		# X is (n x d)
		# Returns (n x n) output Z where
		# Z[i,j] = function(x[i,:],x[j,:])
		# function is a callable like this: 
		# function(x,y)
		K = np.ones((X.shape[0],X.shape[0]))
		for i in range(X.shape[0]): 
			for j in range(i,X.shape[0]):
				k = function(X[i,:],X[j,:])
				K[i,j] = k
				K[j,i] = k
		return K

	def query(self,q,function):
		output = 0.0
		for weight,point in zip(self.beta,self.X_public):
			output += weight*function(point,q)
		output /= np.sum(self.beta)
		return output


	def _compute_basis(self, K_zz):
		""" Compute orthonormal basis of H_M using Gram-Schmidt """

		# Compute the basis in the form of scalars A_{fm}, where
		# A_{fm} is the coefficient of k(z_m, .) in the expansion of e_f
		EPS = 1e-5
		M = len(K_zz)
		A = np.zeros((M, M))
		squared_norms = np.zeros(M)
		K_zz_A = np.zeros((M, M))

		# Step 1: Gram-Schmidt ortogonalization
		for f in range(M):
			A[f][f] = 1
			for j in range(f):
				# project k(z_f, .) onto k(b_j, .) and get the coefficients of z's
				# proj_{b_j}(z_f) = < k(z_f, .), b_j > / norm(b_j) * b_j
				# b_j = A[j][0] k(z_0,.) + ... + A[j][j] k(z_j,.)
				denominator = squared_norms[j]
				if denominator > 0 + EPS:
					# Non-modified GS: compute indicator(f) . K_zz . A[j]
					#numerator = K_zz_A[j][f]
					# Modified GS (more stable): compute A[f] . K_zz . A[j]
					numerator = A[f].dot(K_zz_A[j])
					A[f] -= numerator / denominator * A[j]

			# Compute the pre-multiplication by the K_zz
			K_zz_A[f] = K_zz.dot(A[f])

			# Compute the squared norm of the constructed vector
			squared_norms[f] = A[f].dot(K_zz.dot(A[f]))

			# Print progress
			# if ((f+1) % 100 == 0) or (f+1 == M):
				# print_progress('[Progress] f = %d' % (f+1))
		# print('')

		# Step 2: Gram-Schmidt normalisation
		for f in range(M):
			squared_norm = squared_norms[f]
			if squared_norm > 0 + EPS:
				A[f] /= np.sqrt(squared_norm)

		# Filter non-zero vectors, to obtain basis
		B = []
		parents = []
		for f in range(M):
			if squared_norms[f] > 0 + EPS:
				B.append(A[f])
				parents.append(f+1) # 1-based indexing for parents
		B = np.array(B)

		return B, parents


	def _reweight_public_compute_alpha(self, K_zx_rmeans, K_zz):
		# Compute basis of the RKHS subspace spanned by feature maps of public data points
		B, parents = self._compute_basis(K_zz)

		# Project empirical KME onto the computed basis
		alpha = B.dot(K_zx_rmeans)

		return B, alpha, parents






class KMEReleaseDP_2():
	# Specify your own public synthetic points
	def __init__(self, epsilon, M, data, function, loc = None, sigma_public = 500, debug = False):
		self.debug = debug
		self.epsilon = epsilon
		self.d = data.shape[1] # dimensions
		self.N_private = data.shape[0] # N
		# self.maxM = M
		self.M = M

		# 1. Generate synthetic points
		# sigma_public = 500.0 # default from paper
		X_private = data
		if loc is None: 
			loc = np.zeros(self.d)
		self.X_public = np.random.normal(loc=loc, scale=sigma_public, size=(self.M, self.d))

		# 2. Create kernel matrix means
		# Not actually necessary
		# self.K_xx = self._kernel_matrix(X_private, function)
		# self.K_xx_mean = np.mean(self.K_xx)

		# 3. Get cross kernel means
		self.K_zx_rmeans = self._cross_kernel_matrix_rmeans(self.X_public, X_private, function)

		# 4. Get public kernel matrix
		self.K_zz = self._kernel_matrix(self.X_public, function)

		# 5. Find the basis (B) and weights (alpha) for each data point
		self.B, self.alpha, self.parents = self._reweight_public_compute_alpha(self.K_zx_rmeans, self.K_zz)

		# set_m(): 
		num_dimensions = bisect(self.parents, self.M)
		self.alpha_part = self.alpha[:num_dimensions]
		self.B_part = self.B[:num_dimensions, :self.M]
		self.K_zz_part = self.K_zz[:self.M, :self.M]
		self.K_zx_rmeans_part = self.K_zx_rmeans[:self.M]

		# set_epsilon(): 
		sensitivity = 2.0 / np.sqrt(self.N_private)
		self.beta = self.alpha + np.random.laplace(scale = sensitivity / epsilon, size=len(self.alpha))

		# Now, the final output is self.X_public, self.beta
		# Each point in self.X_public is weighted by a value self.beta

	def set_m(self, m): 
		num_dimensions = bisect(self.parents, self.M)

		self.alpha_part = self.alpha[:num_dimensions]
		self.B_part = self.B[:num_dimensions, :self.M]
		self.K_zz_part = self.K_zz[:self.M, :self.M]
		self.K_zx_rmeans_part = self.K_zx_rmeans[:self.M]
		self.set_epsilon(self.epsilon)

	def set_epsilon(self, epsilon): 
		# It is really expensive to re-compute the kernel matrices all the time
		# so we can just reset epsilon and M without having to recompute
		# using set_epsilon
		# Then, all queries after calling this function will be at this level of epsilon
		# at least until you reset epsilon again
		self.epsilon = epsilon
		sensitivity = 2.0 / np.sqrt(self.N_private)
		self.beta = self.alpha + np.random.laplace(scale = sensitivity / self.epsilon, size=len(self.alpha))

	def _cross_kernel_matrix(self, X, Y, function):
		# X is (n x d) and Y is (m x d)
		# Returns (n x m) output Z where 
		# Z[i,j] = function(x[i,:],y[j,:])
		K = np.zeros((X.shape[0],Y.shape[0]))
		for i in range(X.shape[0]): 
			for j in range(Y.shape[0]): 
				K[i,j] = function(X[i,:],Y[j,:])
		return K

	def _cross_kernel_matrix_rmeans(self, X, Y, function): 

		M,N = len(X), len(Y)
		compute_threshold = 10000 ** 2
		if N*M <= compute_threshold: 
			K_zx = self._cross_kernel_matrix(X, Y, function)
			return np.mean(K_zx, axis = 1)
		else: 
			sums = np.zeros(M)
			step = compute_threshold // M 
			start = 0 
			while start < N: 
				sums += np.sum(self._cross_kernel_matrix(X, Y[start:(start+step)], function), axis=1)
				start = min(start + step, N)
				if self.debug: 
					print("Kernel matrix row means: ",100.0 * start / N)
					sys.stdout.flush()
			return sums / N 
		# self.K_zx = self._cross_kernel_matrix(self.X_public, X_private, function)
		# self.K_zx_rmeans = np.mean(self.K_zx, axis = 1)


	def _kernel_matrix(self, X, function):
		# X is (n x d)
		# Returns (n x n) output Z where
		# Z[i,j] = function(x[i,:],x[j,:])
		# function is a callable like this: 
		# function(x,y)
		K = np.ones((X.shape[0],X.shape[0]))
		for i in range(X.shape[0]): 
			for j in range(i,X.shape[0]):
				k = function(X[i,:],X[j,:])
				K[i,j] = k
				K[j,i] = k
		return K

	def query(self,q,function):
		output = 0.0
		for weight,point in zip(self.beta,self.X_public):
			output += weight*function(point,q)
		output /= np.sum(self.beta)
		return output


	def _compute_basis(self, K_zz):
		""" Compute orthonormal basis of H_M using Gram-Schmidt """

		# Compute the basis in the form of scalars A_{fm}, where
		# A_{fm} is the coefficient of k(z_m, .) in the expansion of e_f
		EPS = 1e-5
		M = len(K_zz)
		A = np.zeros((M, M))
		squared_norms = np.zeros(M)
		K_zz_A = np.zeros((M, M))

		# Step 1: Gram-Schmidt ortogonalization
		for f in range(M):
			A[f][f] = 1
			for j in range(f):
				# project k(z_f, .) onto k(b_j, .) and get the coefficients of z's
				# proj_{b_j}(z_f) = < k(z_f, .), b_j > / norm(b_j) * b_j
				# b_j = A[j][0] k(z_0,.) + ... + A[j][j] k(z_j,.)
				denominator = squared_norms[j]
				if denominator > 0 + EPS:
					# Non-modified GS: compute indicator(f) . K_zz . A[j]
					#numerator = K_zz_A[j][f]
					# Modified GS (more stable): compute A[f] . K_zz . A[j]
					numerator = A[f].dot(K_zz_A[j])
					A[f] -= numerator / denominator * A[j]

			# Compute the pre-multiplication by the K_zz
			K_zz_A[f] = K_zz.dot(A[f])

			# Compute the squared norm of the constructed vector
			squared_norms[f] = A[f].dot(K_zz.dot(A[f]))

			# Print progress
			# if ((f+1) % 100 == 0) or (f+1 == M):
				# print_progress('[Progress] f = %d' % (f+1))
		# print('')

		# Step 2: Gram-Schmidt normalisation
		for f in range(M):
			squared_norm = squared_norms[f]
			if squared_norm > 0 + EPS:
				A[f] /= np.sqrt(squared_norm)

		# Filter non-zero vectors, to obtain basis
		B = []
		parents = []
		for f in range(M):
			if squared_norms[f] > 0 + EPS:
				B.append(A[f])
				parents.append(f+1) # 1-based indexing for parents
		B = np.array(B)

		return B, parents


	def _reweight_public_compute_alpha(self, K_zx_rmeans, K_zz):
		# Compute basis of the RKHS subspace spanned by feature maps of public data points
		B, parents = self._compute_basis(K_zz)

		# Project empirical KME onto the computed basis
		alpha = B.dot(K_zx_rmeans)

		return B, alpha, parents





