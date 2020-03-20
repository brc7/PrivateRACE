from baselines.spectral import * 
from baselines.KMERelease import * 
from baselines.bernstein import * 
from baselines.PFDA import * 

from race.race import *
from race.hashes import *

import numpy as np 
import matplotlib.pyplot as plt 

# Evaluation of all methods on 1D salary datasets
# (Some methods do not scale well to high dimensions) 

# Define KDE and kernel function
bandwidth = 5000
scale_factor = 250000

def kernel(x,y): 
    return P_L2(np.linalg.norm(x-y),bandwidth)

def ScaledKernel(x,y): 
    return P_L2(np.linalg.norm(x-y),bandwidth/scale_factor)


def KDE(x,data): 
	# KDE for this choice of kernel and bandwidth
	val = 0
	n = 0
	for xi in data: 
		val += kernel(x,xi)
		n += 1
	return val / n

def ScaledKDE(x,data): 
	val = 0
	n = 0 
	for xi in data: 
		val += ScaledKernel(x,xi)
		n += 1
	return val / n 



############################################
# Data scientist salaries
############################################

salaries = np.loadtxt('data/sf_employee_2018.csv',delimiter = '\n')
# salaries = np.loadtxt('data/nyc_rs.csv',delimiter = '\n')
N = len(salaries)
print(N)
q = np.linspace(10000,250000,200)
epsilon = 1.0 

# # Ground Truth KDE
# gtruth = np.zeros_like(q)
# for i,qi in enumerate(q): 
#     gtruth[i] = KDE(qi,salaries)
# plt.plot(q,gtruth, 'k', linewidth = 2.0, label = 'KDE')


# # KME 
# kmeM = 80
# kmeSalaries = np.reshape(salaries,(N,1))
# kmeAlgo = KMEReleaseDP_2(epsilon, kmeM, kmeSalaries, kernel, loc = 100000, sigma_public = 50000, debug = True)
# kme_results = np.zeros_like(q)
# for i,qi in enumerate(q):
# 	kme_results[i] = kmeAlgo.query(qi,kernel)
# plt.plot(q,kme_results, label = 'KME')

# # Bernstein 
# scaled_data = np.reshape(salaries*1.0/scale_factor,(N,1))
# bernsteinM = 50
# bernsteinAlgo = ScalableBernsteinDP(epsilon, bernsteinM, scaled_data, ScaledKDE, 1.0 / N)
# bernstein_results = np.zeros_like(q)
# for i,qi in enumerate(q):
# 	bernstein_results[i] = bernsteinAlgo.query([qi*1.0/scale_factor])
# plt.plot(q,bernstein_results, label = 'Bernstein')

# Spectral
# spectralM = 2
# scaled_data = np.reshape(salaries*1.0/scale_factor,(N,1))
# spectralAlgo = ScalableSpectralDP(epsilon, spectralM, scaled_data, ScaledKDE, 1.0 / N)
# spectral_results = np.zeros_like(q)
# for i,qi in enumerate(q):
# 	spectral_results[i] = spectralAlgo.query(qi/scale_factor)
# plt.plot(q,spectral_results, label = 'Spectral')

# PFDA 
curves = [[ScaledKernel(xi,qi) for qi in q] for xi in salaries]
curves = np.array(curves)
def g(x,y): 
	return np.exp(-np.linalg.norm(x - y)**2 / 0.01 )
pfdaAlgo = PFDA(epsilon, curves, g, delta = 0.01)
plt.plot(q,pfdaAlgo.f_tilda,label = 'PFDA')
np.savetxt("PFDA-f-tilda.txt", pfdaAlgo.f_tilda, delimiter=',')
np.savetxt("PFDA-f.txt", pfdaAlgo.f, delimiter=',')
np.savetxt("PFDA-Z.txt", pfdaAlgo.Z, delimiter=',')
print(pfdaAlgo.d2)

# RACE
# reps = 200
# hash_range = 1000
# lsh = L2LSH(reps,1,bandwidth)

# S = RACE(reps, hash_range)
# for xi in salaries: 
# 	S.add(lsh.hash(xi))
# S.set_epsilon(epsilon)

# race_results = np.zeros_like(q)
# for i,qi in enumerate(q):
# 	race_results[i] = S.query(lsh.hash(qi))
# plt.plot(q,race_results, label = 'RACE')



plt.legend()
plt.show()



