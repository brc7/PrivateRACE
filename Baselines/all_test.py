import numpy as np
import scipy.stats
import matplotlib.pyplot as plt 
import sys 

from bernstein import * 
from spectral import * 


def KDE(x,data): 
	val = 0
	n = 0 
	for xi in data: 
		val += np.exp(-np.linalg.norm(x - xi) / 0.2 )
		n += 1
	return val / n 

N = 1000
d = 1
np.random.seed(42) # lol
data1 = np.random.normal(loc = 0.2,scale = 0.1,size = (N,d))
data2 = np.random.normal(loc = 0.8,scale = 0.02,size = (N,d))
data = np.concatenate((data1,data2),axis = 0)

algo1 = BernsteinDP(0.5, 40, data, KDE, 1.0 / N)
algo2 = SpectralDP(0.5, 4, data, KDE, 1.0 / N)

q = np.linspace(0,1,100)
q = np.reshape(q,(100,1))

real_out = []
out_1 = []
out_2 = []

for qi in q: 
	real_out.append(KDE(qi,data))
	out_1.append(algo1.query(qi))
	out_2.append(algo2.query(qi))


plt.figure()
plt.plot(q,real_out,label = "Real KDE")
plt.plot(q,out_1, '--',label = "Private Bernstein Approximation")
plt.plot(q,out_2, '-.',label = "Private Spectral Approximation")
plt.title("Orthogonal Basis Function Release")
plt.legend()
plt.show()