from baselines.spectral import * 
from baselines.KMERelease import * 
from baselines.bernstein import * 
from race.race import *
from race.hashes import *

import numpy as np 
import matplotlib.pyplot as plt 
import sys
import math



# Load dataset
data = np.loadtxt('src/synthetic.csv',delimiter = ',')
data = data[0:1000:]
data = data[:,0:2]


print(data.shape)
N = data.shape[0]
d = data.shape[1]

# Generate queries
np.random.seed(420)
NQ = 400
queries = np.random.random(size = (NQ,d))
queries -= 0.5
queries *= min(data.max(),np.abs(data.min()))


# Generate ground truth
print("Computing ground truth for",NQ,"queries")
sys.stdout.flush()

def k(x,y): 
	# w is GLOBAL (sorry, so so sorry)
	# i had to get this done so i could sleep ok? ok :(
	w = 5.0
	return P_L2(np.linalg.norm(x - y),w)

def KDE(x,data): 
	val = 0
	n = 0 
	for xi in data:
		val += k(xi,x)
		n += 1
	return val / n 

gt = []
for qi in queries: 
	gt.append(KDE(qi,data))
	print('',end = ".")
	sys.stdout.flush()
gt = np.array(gt)
print("\nGround truth finished")
sys.stdout.flush()



# Now compute the actual DP approximants
'''
epsilon = np.linspace(0.001,5,20)

# 1. Compute KME release results
A_KME = KMEReleaseDP_1(1.0, 200, data, k, sigma_public = 50.0, debug = True)
KME_Results = []

for ep in epsilon: 
	print("Attempting KME with epsilon =",ep)
	sys.stdout.flush()
	A_KME.set_epsilon(ep)
	vals = []
	
	for qi in queries: 
		vals.append(A_KME.query(qi,k))
	vals = np.array(vals)

	KME_Results.append([np.mean(np.abs(vals - gt)),np.std(np.abs(vals - gt))])

KME_Results = np.array(KME_Results)
print("KME Method")
print(epsilon)
print(KME_Results)


# plt.plot(epsilon,KME_Results[:,0])
plt.errorbar(epsilon,KME_Results[:,0],KME_Results[:,1],label = "KME")

# plt.show()
# '''




# '''
# 1. Compute bernstein results 
epsilon = np.geomspace(0.1,2,10)
BS_Results = []

dmin = np.min(data)
dmax = np.max(data)

data_BS = data - dmin
data_BS /= (dmax - dmin)

queries_BS = queries - dmin
queries_BS /= (dmax - dmin)
# w = sigma*1.0 / (dmax - dmin) # tbh idk what this is for

A_BS = ScalableBernsteinDP(1.0, 4, data_BS, KDE, 1.0 / N, debug = False)

for ep in epsilon: 
	print("Attempting bernstein with epsilon =",ep)
	sys.stdout.flush()
	A_BS.set_epsilon(ep)

	vals = []
	for qi in queries_BS: 
		vals.append(A_BS.query(qi))
	vals = np.array(vals)
	print(np.mean(np.abs(vals - gt)))
	BS_Results.append( (np.mean(np.abs(vals - gt)),np.std(np.abs(vals - gt))))

BS_Results = np.array(BS_Results)
plt.errorbar(epsilon,BS_Results[:,0],BS_Results[:,1],label = "BS")
# plt.plot(epsilon,BS_Results[:,0])

print("BS Method")
print(epsilon)
print(BS_Results)



# '''




# algo = KMEReleaseDP_1(epsilon,50, data, k, sigma_public = 1.0)
# algo = BernsteinDP(epsilon, 40, data, KDE, 1.0 / N)

# kme with KMEReleaseDP_1(1.0, 100, data, k, sigma_public = 500.0)
# [[0.027758674891088107, 0.0013396182997601111], [0.027703628175906589, 0.0013396624457882185], [0.027744418642216443, 0.0013396978121366734], [0.027540107295062621, 0.0013407320184112925], [0.027606654683846505, 0.0013413184939445393], [0.02772829663597947, 0.0013397801991502378], [0.027688148686485375, 0.0013397089785455001], [0.027761924670997772, 0.0013397255616279368], [0.02787898457145925, 0.0013392247165418267], [0.02761782059565223, 0.0013393519861465153], [0.027713101058968993, 0.0013398210793269221], [0.027681679796939727, 0.0013397811037315422], [0.02773379435997704, 0.0013397643965981773], [0.0276531335746804, 0.0013396187941385413], [0.02773210989687179, 0.0013397922109765529], [0.027709680518023729, 0.0013398860769270812], [0.027668681843227551, 0.0013389770067771798], [0.027714685889828856, 0.0013397921284830491], [0.027833063857253419, 0.0013395059319084755], [0.027761706606856104, 0.0013395845540524122], [0.027753399285826531, 0.0013397624664981877], [0.027701594631536926, 0.0013397819074444492], [0.027732884001986462, 0.0013397566095707592], [0.026880398258730853, 0.0013416772552214376], [0.027693097121410725, 0.0013397517952087761], [0.027739326452744715, 0.0013397900540711161], [0.027717135286002677, 0.0013398213767899837], [0.027897803574469467, 0.0013396346167732439], [0.027771406710813674, 0.0013397868189889005], [0.027686632008369204, 0.0013397582219694972], [0.02771374548350241, 0.0013396677457175328], [0.027730292395353059, 0.0013397163873708213], [0.02772136266675158, 0.0013399230342965147], [0.027836218976437621, 0.0013396596248588675], [0.027744241987409733, 0.001339758634286297], [0.027761424605850436, 0.0013398166164457622], [0.027489527046754933, 0.001339993938985824], [0.027721835966209876, 0.0013398764688285986], [0.027740884270315673, 0.0013396809569784685], [0.027755340823242346, 0.0013396907408994168], [0.027704317422497158, 0.0013398164086928408], [0.027742231632732314, 0.0013397405580141538], [0.02769587825688178, 0.0013397491673597748], [0.027713103863268206, 0.0013398544370477757], [0.028934269786606234, 0.0013397142280328009], [0.027736413141677699, 0.0013398150532518672], [0.027745184686595682, 0.0013396363169358965], [0.027725004137560929, 0.0013397568668526284], [0.027668721287477126, 0.0013398659875128094], [0.027736531591361614, 0.0013396549220785611]]

plt.legend()
plt.show()
