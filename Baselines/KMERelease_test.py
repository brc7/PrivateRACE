import numpy as np
import scipy.stats
import matplotlib.pyplot as plt 
import sys 

from KMERelease import * 

def KDE(x,data): 
	val = 0
	n = 0 
	for xi in data: 
		val += np.exp(-np.linalg.norm(x - xi) / 0.2 )
		n += 1
	return val / n 

def k(x,y): 
	return np.exp(-np.linalg.norm(x - y) / 0.2 )

# bias does not depend on N 
N = 1000
d = 1
np.random.seed(42) # lol
data1 = np.random.normal(loc = -0.5,scale = 0.1,size = (N,d))
data2 = np.random.normal(loc = 0.2,scale = 0.02,size = (N,d))
data = np.concatenate((data1,data2),axis = 0)

# def __init__(self, epsilon, M, data, function, debug = False):

np.random.seed(20)
algo = KMEReleaseDP_1(1.0,50, data, k, sigma_public = 1.0, debug = True)

q = np.linspace(-1,1,100)
q = np.reshape(q,(100,1))

real_out = []
fake_out = []
for qi in q: 
	real_out.append(KDE(qi,data))
	fake_out.append(algo.query(qi,k))

plt.figure()
plt.plot(q,real_out,label = "Real KDE")
plt.plot(q,fake_out, label = "KME Release Approximation")
plt.legend()
plt.show()
