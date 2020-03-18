import numpy as np
import scipy.stats
import matplotlib.pyplot as plt 
import sys 

from PFDA import * 


def g(x,y): 
	return np.exp(-np.linalg.norm(x - y) / 0.2 )

def KDE(x,data): 
	val = 0
	n = 0 
	for xi in data: 
		val += g(x,xi)
		n += 1
	return val / n 

def k(x,y): 
	return np.exp(-np.linalg.norm(x - y)**2 / 0.01 )

N = 100
d = 1
np.random.seed(42) # lol
data = np.random.normal(loc = -0.2,scale = 0.1,size = (N,d))


yy = np.linspace(-1,1,100)
# curves = [[k(x,y) for y in yy] for x in data]
# curves = [[g(x,y) for y in yy] for x in data]
curves = [[g(x,y) for y in yy] for x in data]


curves = np.array(curves)

gt = []
for q in yy: 
	gt.append(KDE(q,data))


np.random.seed(20)
algo = PFDA(1.0, curves, k)


plt.figure()
for curve in curves: 
	plt.plot(yy,curve,'k-',alpha = 0.1)
plt.plot(yy,algo.f,linewidth = 2,label = "Private Function Mean")
plt.plot(yy,gt, label = "Ground Truth KDE")
plt.legend()
plt.show()
