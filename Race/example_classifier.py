import numpy as np
import matplotlib.pyplot as plt 

from race import * 
from hashes import * 


sigma = 5.0 # bandwidth of KDE
epsilon = 1 # privacy parameter
dimensions = 2 

# RACE sketch parameters
reps = 80
hash_range = 50

lsh = L2LSH(reps,dimensions,sigma)
S = RACE(reps, hash_range)

# Generate data
xmin,xmax = (-50,50)
ymin,ymax = (-50,50)
N = 500

cov1 = np.array([[60,0],[0,60]])
cov2 = np.array([[50,30],[30,60]])
D1 = np.random.multivariate_normal([-10,-8], cov1, N)
D1b = np.random.multivariate_normal([0,20], cov2, N)
D1 = np.vstack((D1,D1b))

D2 = np.random.multivariate_normal([15,15], cov2, N)
D2b = np.random.multivariate_normal([30,0], cov1, N)
D2 = np.vstack((D2,D2b))


for d in D2:
	S.add(lsh.hash(d))
for d in D1: 
	S.remove(lsh.hash(d))


# Make the sketch private
S.make_private(epsilon)

# Plot the decision regions for the private max likelihood classifier
M = 100
x = np.linspace(xmin,xmax,M)
y = np.linspace(ymin,ymax,M)
Z = np.zeros((M,M))

for i,xi in enumerate(x): 
	for j,yi in enumerate(y): 
		Z[j,i] = S.query(lsh.hash(np.array([xi,yi])))

X,Y = np.meshgrid(x,y)
cs = plt.contourf(X, Y, Z, 15, cmap=plt.cm.RdBu_r,vmax=np.max(Z), vmin=np.min(Z))
plt.plot(D1[:,0],D1[:,1],'ks', fillstyle = 'none', markersize = 4)
plt.plot(D2[:,0],D2[:,1],'k^', fillstyle = 'none', markersize = 4)
level = ( np.max(Z) + np.min(Z) )/2.0
plt.contour(X, Y, Z, [level], colors = ['k'])

plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.show()
