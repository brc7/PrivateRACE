import numpy as np
import scipy as sp 

'''
Approach does not directly apply to higher dimensions

Approach is to return a mean of FUNCTIONS

You give it a bunch of functions, it returns the mean


'''


class PFDA():
	def __init__(self, epsilon, data, function, debug = False):
		self.epsilon = epsilon
		# dimensions = curve values
		phi = 0.01

		# 0. Create grid 
		# grid=seq(0,1,length.out = dim(Data)[1])
	    # n=length(grid)
		grid = np.linspace(0,1,data.shape[0])
		n = len(grid)

		# 1. Create gram matrix
		# Sig=matrix(nrow=n,ncol=n)   # covariance matrix in the grid [0,1]
		# for(i in 1:n){
		# 	for(j in 1:n){  # Sig_{i,j}=C( (i-1)/(n-1) , (j-1)/(n-1) )
		# 		Sig[i,j] = C(grid[i],grid[j],ro=ro)
		self.sig = self._kernel_matrix(np.reshape(grid,(n,1)), function)

		# 2. Take the eigenvalues of the gram matrix
		# gamma=eigen(Sig)$values
		# e.val.z=gamma[1:n]/n
		# e.vec.z=eigen(Sig)$vectors[,1:n]
		# m=length(which(e.val.z>0))
		eigenvalues,eigenvectors = np.linalg.eig(self.sig)
		self.e_val_z = np.real(eigenvalues) * 1.0 / n
		self.e_vec_z = np.real(eigenvectors)
		m = len(np.where(self.e_val_z > 0))

		# 3. Generating Gaussian proccess Z based on KL-expansion

		# Z=matrix(0,nrow = n,ncol = 1)
		# for(i in 1:m){
		# Z=Z+sqrt(e.val.z[i])*rnorm(1)*e.vec.z[,i]
		# }
		self.Z = np.zeros(n)
		for i in range(m):
			self.Z += np.sqrt(self.e_val_z[i])*np.random.normal() * self.e_vec_z[:,i]

		# 4. Generate f_D = \mu hat
		# xg=Data
		# N=dim(xg)[2]
		# Ym=matrix(NA,n,N)
		# for(s in 1:N){ # for each curve in the set of curves
		# Y=matrix(0,n,1) # create a projected version of this curve
		# for(i in 1:m){ # for each eigenvalue
		#   Y=Y+(e.val.z[i]/(e.val.z[i]+phi))*
		#     (as.numeric(t(xg[,s]))%*%e.vec.z[,i])*e.vec.z[,i]
		# }
		# Ym[,s]=Y
		# }
		# f=rowMeans(x = Ym)
		N = data.shape[1]
		Ym = np.zeros((n,N))
		for s in range(N):
			Y = np.zeros(n)
			for i in range(m):
				# Inner product of this curve with the other curves
				Y = Y + self.e_val_z[i] / (self.e_val_z[i] + phi) * np.inner(data[:,s],self.e_vec_z[:,i]) * self.e_vec_z[:,i]
			Ym[:,s] = Y
			print(np.linalg.norm(data[:,s]-Y))
		self.f = np.mean(Ym,axis = 0)

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

