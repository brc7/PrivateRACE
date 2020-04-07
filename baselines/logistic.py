import numpy as np 
import sys 
# Implementation of "Privacy-preserving logistic regression" in NIPS09
# credit to Martin Pellarolo for his Medium blog post "Logistic Regression from scratch in Python"
# which I adapted to produce this code

class PrivateLogisticRegression:
	def __init__(self, fit_intercept = True, debug = False):
		self.fit_intercept = fit_intercept
		self.debug = debug
		# 1. Pick random vector b from density h(b) ~ e^−epsilon/||b||
		# To implement this, pick the norm of b from gamma (d,2/epsilon) distribution
		# and the direction uniform at random

	def __random_unit_vector(self, d):
		b = np.random.normal(size = d)
		while np.linalg.norm(b) < 10e-8:
			b = np.random.normal(size = d)
		return b / np.linalg.norm(b)

	def __b(self, d, epsilon):
		magnitude = np.random.gamma(d, 2.0/epsilon)
		direction = self.__random_unit_vector(d)
		return magnitude * direction

	def __add_intercept(self, X):
		intercept = np.ones((X.shape[0], 1))
		return np.concatenate((intercept, X), axis=1)

	def __sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def __loss(self, w, lam, b, X, y):
		n = X.shape[0]
		inner_products = np.dot(X, w)
		z = self.__sigmoid(inner_products)
		logistic_loss = y*np.log(z) + (1 - y)*np.log(1 - z)
		# 0.5*lam*np.dot(w,w) + np.dot(b,w)*1.0/n 
		return 0.5*lam*np.dot(w,w) + np.dot(b,w)*1.0/n - np.mean(logistic_loss,axis = 0)


	def __gradient(self, w, lam, b, X, y):
		n = X.shape[0]
		logistic_grad = np.multiply(self.__sigmoid(np.dot(X,w)) - y, X.T).T
		# lam*w + b/n +
		return lam*w + b/n + np.mean(logistic_grad,axis = 0)

	def fit(self, X, y, epsilon, lam = 0.1, eta = 1.0, iters = 10000):
		if self.fit_intercept:
			X = self.__add_intercept(X)
		
		# weights initialization
		self.w = np.zeros(X.shape[1])
		b = self.__b(X.shape[1], epsilon)

		for i in range(iters):
			grad = self.__gradient(self.w, lam, b, X, y)
			self.w -= eta*grad
			if(self.debug == True and i % 100 == 0):
				print(f'loss: {self.__loss(self.w, lam, b, X, y)} Gradient norm: {np.linalg.norm(grad)}\t')
				sys.stdout.flush()

	
	def predict_prob(self, X):
		if self.fit_intercept:
			X = self.__add_intercept(X)

		return self.__sigmoid(np.dot(X, self.w))
	
	def predict(self, X, threshold):
		return self.predict_prob(X) >= threshold



