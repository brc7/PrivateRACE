import numpy as np
import scipy.stats
import matplotlib.pyplot as plt 
import sys 
import pickle
from bernstein import * 

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
# data = np.linspace(0,1,N)
# data = np.reshape(data,(N,1))
print(data.shape)
algo = ScalableBernsteinDP(10.0, 40, data, KDE, 1.0 / N)

# handle = open('temp.pickle', 'wb')
# pickle.dump(algo_1, handle, protocol=pickle.HIGHEST_PROTOCOL)
# handle = open('temp.pickle', 'rb')
# algo = pickle.load(handle)

q = np.linspace(0,1,100)
q = np.reshape(q,(100,1))

real_out = []
fake_out = []
for qi in q: 
	real_out.append(KDE(qi,data))
	fake_out.append(algo.query(qi))


plt.figure()
plt.plot(q,real_out,label = "Real KDE")
plt.plot(q,fake_out, label = "Bernstein Approximation")
plt.legend()
plt.show()
sys.exit()

N = 100
d = 2
np.random.seed(42) # lol
data = np.random.multivariate_normal((0.5,0.5), ((0.01,0.02),(0,0.01)), size = N)
print(data.shape)
algo_1 = BernsteinDP(1000, 20, data, KDE, 1.0 / N)


M = 20
t1 = np.linspace(0,1,M)
t2 = np.linspace(0,1,M)
T1,T2 = np.meshgrid(t1,t2)
Z = np.zeros((M,M))
Zp = np.zeros((M,M))
for i,t1i in enumerate(t1):
	print('.',end='')
	sys.stdout.flush()
	for j,t2j in enumerate(t2):
		Z[i,j] = KDE(np.array([t1i,t2j]),data)
		Zp[i,j] = algo.query(np.array([t1i,t2j]))

plt.figure()
plt.subplot(121)
plt.title("Ground truth KDE")
cs = plt.contourf(T2, T1, Z, 100, vmax=1, vmin=0)
plt.subplot(122)
plt.title("Bernstein KDE")
cs = plt.contourf(T2, T1, Zp, 100,vmax=1, vmin=0)

plt.show()

