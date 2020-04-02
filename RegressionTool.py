import numpy as np 
import matplotlib.pyplot as plt 
import pickle

from race.race import *
from race.hashes import *
from race.optimization import *


from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import sys
import random
import pickle
import os 

# for diabetes: 200 iters, eta = 0.5, sigma = 0.5, beta = 0.5, 8 components 
# for naval: 500 iters, eta = 0.1, all same as above 

# testfile = 'data/regression/diabetes/diabetes_validation.csv'
# datafile = 'data/regression/diabetes/diabetes.csv'
testfile = 'data/regression/naval/naval_validation.txt'
datafile = 'data/regression/naval/naval.txt'

reps = 8000
p = 4

sigma = 0.5
eta = 0.1

n_iters = 500
beta = 0.5
n_components = 8

use_prp = True
use_intercept = True

n_experiment_repetitions = 10

# epsilon = np.geomspace(0.001,1.0,10)
epsilon = [0.00001,0.00002,0.00005,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10]


def max_scale(x,y,xv,yv): # train features, train y, test features, test y
	# scale based on the max abs components
	sx = np.max(abs(x), axis=0) # scaling factor x : sx 
	sy = np.max(abs(y))
	x /= sx
	y /= sy
	xv /= sx
	yv /= sy
	return (x,y,xv,yv)

def load(datafile,testfile):
	name = os.path.splitext(os.path.basename(datafile))[0]
	if name == "diabetes": 
		X = np.loadtxt(datafile,delimiter = ',')
		Xv = np.loadtxt(testfile,delimiter = ',')
		y = X[:,-1]
		x = X[:,:-1]
		xv = Xv[:,:-1] # x test (xv)
		yv = Xv[:,-1] # y test (yv)
		return max_scale(x,y,xv,yv)
	if name == "naval": 
		X = np.loadtxt(datafile)
		Xv = np.loadtxt(testfile)
		x = X[:, :-2]
		y = X[:,-1]
		xv = Xv[:,:-2] # x test (xv)
		yv = Xv[:,-1] # y test (yv)
		return max_scale(x,y,xv,yv)
	if name == "indoor":
		X = np.loadtxt(datafile,delimiter = ',')
		Xv = np.loadtxt(testfile,delimiter = ',')
		x = X[:, :-9]
		y = X[:,  -8] # latitude and longitude in -9 and -8
		xv = Xv[:, :-9]
		yv = Xv[:,  -8]

		y_offset = np.min(y)
		x_offset = np.min(x,axis = 0)
		y -= y_offset
		x -= x_offset
		xv -= x_offset
		yv -= y_offset
		return max_scale(x,y,xv,yv)






x,y,x_test,y_test = load(datafile,testfile)


testset = format_dataset(x_test,y_test,intercept = use_intercept)
dataset = format_dataset(x,y,intercept = use_intercept)
N = testset.shape[0]
d = testset.shape[1]

# Construct LSH function
np.random.seed(42)
LSH = FastSRPMulti(reps,d,p)


# Construct RACE sketch
filename = os.path.splitext(datafile)[0]+'-RACE-'+str(p)+'-'+str(reps)+'.pickle'
if os.path.isfile(filename):
	with open(filename, 'rb') as handle: 
		S = pickle.load(handle)
else:
	S = RACE(reps,2**p)
	S = construct_race_sketch(dataset, LSH, S, prp = use_prp, verbose = True)
	with open(filename, 'wb') as handle: 
		pickle.dump(S, handle, protocol=pickle.HIGHEST_PROTOCOL)

def R0(theta): 
	return regression_loss(theta,testset)


# Sanity check baselines 
baseline_model = optimal_linregress(x,y,intercept = use_intercept)
m = np.append(baseline_model,-1)
m = np.reshape(m,(d,1))
optimal_loss = regression_loss(m,testset) * 1.0/N
trivial_loss = np.sum(y_test**2)*1.0/N
mean_loss = np.sum( (y_test - np.mean(y))**2 )*1.0/N
print("Optimal Loss: ",optimal_loss)
print("Trivial Loss: ",trivial_loss)
print("Mean Loss: ",mean_loss)

# Sweep over privacy parameter
results = []
for ep in epsilon: 
	errs = np.zeros(n_experiment_repetitions)
	for i in range(n_experiment_repetitions): 
		print("Epsilon = ",ep,end = ',')
		sys.stdout.flush()

		np.random.seed(i)
		S.set_epsilon(ep)
		theta, surrogate_losses, real_losses = accelerated_race_zgd(S, LSH, n_iters, eta, beta, sigma, n_components, 
			verbose = False, loss = None, dual = (not use_prp))

		loss = regression_loss(theta,testset)*1.0/N
		print(" Loss = ",loss)
		sys.stdout.flush()
		errs[i] = loss
	errmean = np.mean(errs)
	errstd = np.std(errs)
	print("Epsilon = ",ep,"Mean = ",np.mean(errs),"STD = ",np.std(errs))
	sys.stdout.flush()
	results.append((errmean,errstd))

print('')
print('RESULTS:')
print('')
print(results)




