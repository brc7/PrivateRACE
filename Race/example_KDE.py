import numpy as np
import matplotlib.pyplot as plt 

from race import * 
from hashes import * 


sigma = 10.0 # bandwidth of KDE
epsilon = 1 # privacy parameter
dimensions = 2 

# RACE sketch parameters
reps = 80
hash_range = 50

lsh = L2LSH(reps,dimensions,sigma)
S = RACE(reps, hash_range)

# Generate data
np.random.seed(420)

N = 500
cov1 = np.array([[60,0],[0,60]])
cov2 = np.array([[40,0],[0,40]])
cov3 = np.array([[40,-30],[-30,40]])
cov4 = np.array([[50,40],[40,60]])
D1a = np.random.multivariate_normal([-30,-30], cov1, N)
D1b = np.random.multivariate_normal([30,30], cov2, N)
D1c = np.random.multivariate_normal([-10,10], cov3, N)
D1d = np.random.multivariate_normal([20,-15], cov4, N)
data = np.vstack((D1a,D1b,D1c,D1d))

# feed data into RACE 
for d in data: 
	S.add(lsh.hash(d))

S.make_private(epsilon)

xmin,xmax = (-80,80)
ymin,ymax = (-80,80)

M = 200
x = np.linspace(xmin,xmax,M)
y = np.linspace(ymin,ymax,M)
Z = np.zeros((M,M))

for i,xi in enumerate(x): 
	for j,yi in enumerate(y): 
		Z[j,i] = S.query(lsh.hash(np.array([xi,yi])))

X,Y = np.meshgrid(x,y)

cs = plt.contourf(X, Y, Z, 15, cmap=plt.cm.RdBu_r,vmax=np.max(Z), vmin=np.min(Z))
plt.plot(data[:,0],data[:,1],'k.', fillstyle = 'none', markersize = 4,alpha = 0.2)
plt.show()
