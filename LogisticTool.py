import numpy as np 
import matplotlib.pyplot as plt 
import pickle

from baselines.logistic import * 
from sklearn.linear_model import LogisticRegression

import sys
import random
import pickle
import os 

testfile = 'data/class/pulsar/pulsar_test.npy'
trainfile = 'data/class/pulsar/pulsar_train.npy'

eta = 10.0
lam = 0
n_iters = 5000
ep = 10e10

n_experiment_repetitions = 10
epsilon = [0.00001,0.00002,0.00005,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10]

testset = np.load(testfile)
trainset = np.load(trainfile)

y_train = trainset[:,0]
X_train = trainset[:,1:]
y_test = testset[:,0]
X_test = testset[:,1:]

# y_train[np.where(y_train==0)[0]] = -1

# clf = LogisticRegression()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(1.0 - np.mean(np.abs(y_pred - y_test)))

results = []


for ep in epsilon: 
	print("Epsilon =",ep)
	errs = np.zeros(n_experiment_repetitions)

	for exp_id in range(n_experiment_repetitions):
		model = PrivateLogisticRegression(debug = False)
		model.fit(X_train, y_train, ep, lam, eta, n_iters)
		y_pred = model.predict(X_test,0.5)
		accuracy = 1.0 - np.mean(np.abs(y_pred - y_test))
		errs[exp_id] = accuracy

		print('Accuracy :',accuracy)
		sys.stdout.flush()

	errmean = np.mean(errs)
	errstd = np.std(errs)
	results.append((errmean,errstd))

	print("Epsilon = ",ep,"Mean = ",np.mean(errs),"STD = ",np.std(errs))
	sys.stdout.flush()

print(results)


