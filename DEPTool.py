import numpy as np 
import argparse
import sys
import os
import time
import pickle

from baselines.spectral import * 
from baselines.KMERelease import * 
from baselines.bernstein import * 
from race.race import *
from race.hashes import *

''' Experiment pre-processing tool
'''

parser = argparse.ArgumentParser(description = "Density Experiment Preprocessing (DEP) tool - preprocess private function summaries")
parser.add_argument("data", help=".npy file with (n x d) data entries")
parser.add_argument("kernel_id", type=int, help="0: L2 LSH kernel, 1: Angular kernel")
parser.add_argument("bandwidth", type=float, help="Density estimate bandwidth")
parser.add_argument("-r","--race", type=int, nargs = 2, help="(range,reps) Prepare RACE with (range,reps) parameters")
parser.add_argument("-b","--bernstein", type=int, nargs = 2, help="Prepare BernsteinDP with a K-lattice")
# parser.add_argument("-s","--spectral", type=int, help="Prepare SpectralDP with a K-lattice")
parser.add_argument("-kme","--kmerelease", type=int, nargs = 2, help="(M,sigma) Kernel mean embedding release with M synthetic points, Gaussian distributed with sigma")
parser.add_argument("-v","--verbose", help="Verbose output",action="store_true")
args = parser.parse_args()


if args.kernel_id == 0: 
	kernel = lambda x,y,w : P_L2(np.linalg.norm(x-y),w)
elif args.kernel_id == 1:
	kernel = lambda x,y,w : P_SRP(x,y)**(int(w))
else: 
	print("Unsupported kernel id.") 
	sys.exit()

def KDE(x,data): 
	# KDE for this choice of kernel and bandwidth
	val = 0
	n = 0
	for xi in data: 
		val += kernel(x,xi,args.bandwidth)
		n += 1
	return val / n


dataset = np.load(args.data)
dataset = dataset[0:100,:]

N,d = dataset.shape

if args.race: 
	# Do RACE 
	hash_range = args.race[0]
	reps = args.race[1]

	if args.kernel_id == 0: 
		lsh = L2LSH(reps,d,args.bandwidth)
	elif args.kernel_id == 1:
		lsh = SRPMulti(reps,d,int(args.bandwidth))
	else: 
		print("Unsupported kernel (hash function) id.") 
		sys.exit()

	print("Preprocessing RACE with range =",hash_range,"reps =",reps)
	sys.stdout.flush()

	start = time.time()
	algo = RACE(reps, hash_range)
	# feed data into RACE
	for i,x in enumerate(dataset): 
		algo.add(lsh.hash(x))
		if i % 1000 == 0: 
			sys.stdout.write('\r')
			sys.stdout.write('Progress: {0:.4f}'.format(i/N * 100)+' %')
			sys.stdout.flush()
	print('')
	sys.stdout.flush()
	end = time.time()

	print("RACE:",end - start,"s")
	print("Saving RACE...")
	sys.stdout.flush()

	# Now save it
	filename = os.path.splitext(args.data)[0]+'RACE-'+str(hash_range)+'-'+str(reps)+'.pickle'
	with open(filename, 'wb') as handle: 
		pickle.dump(algo, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print("RACE saved to",filename)
	sys.stdout.flush()


if args.bernstein: 
	# do scalable bernstein 
	# do KME release
	M = args.bernstein[0]
	S = args.bernstein[1]
	print("Preprocessing Bernstein with M =",M," scale factor = ",S)
	sys.stdout.flush()

	start = time.time()

	scaled_bw = args.bandwidth
	if args.kernel_id == 0:
		scaled_bw = scaled_bw / S

	def ScaledKDE(x,data): 
		# KDE for this choice of kernel and bandwidth
		val = 0
		n = 0
		for xi in data: 
			val += kernel(x,xi,scaled_bw)
			n += 1
		return val / n

	algo = ScalableBernsteinDP(1.0, M, dataset/S, ScaledKDE, 1.0 / N, debug = args.verbose)
	end = time.time()

	print("ScalableBernsteinDP:",end - start,"s")
	print("Saving ScalableBernsteinDP...")
	sys.stdout.flush()

	# Now save it
	filename = os.path.splitext(args.data)[0]+'ScalableBernsteinDP-'+str(M)+'-'+str(S)+'.pickle'
	with open(filename, 'wb') as handle: 
		pickle.dump(algo, handle, protocol=pickle.HIGHEST_PROTOCOL)

	print("ScalableBernsteinDP saved to",filename)
	sys.stdout.flush()

if args.kmerelease:
	# do KME release
	M = args.kmerelease[0]
	sigma = args.kmerelease[1]
	print("Preprocessing KMERelease with M =",M,"sigma =",sigma)
	sys.stdout.flush()

	start = time.time()
	
	kf = lambda x,y : kernel(x,y,args.bandwidth) # special kernel function format for KMERelease
	algo = KMEReleaseDP_1(1.0, M, dataset, kf, sigma_public = sigma, debug = args.verbose)

	end = time.time()
	print("KMEReleaseDP:",end - start,"s")
	print("Saving KMEReleaseDP...")
	sys.stdout.flush()

	# Now save it
	filename = os.path.splitext(args.data)[0]+'KMEReleaseDP-'+str(M)+'-'+str(sigma)+'.pickle'
	with open(filename, 'wb') as handle: 
		pickle.dump(algo, handle, protocol=pickle.HIGHEST_PROTOCOL)

	print("KMEReleaseDP saved to",filename)
	sys.stdout.flush()


# if args.spectral: 
# 	# do scalable spectral
# 	print("Preprocessing SpectralDP with M = ",args.spectral)
# 	sys.stdout.flush()

# 	start = time.time()
# 	M = args.spectral
# 	algo = ScalableSpectralDP(1.0, M, dataset, KDE, 1.0 / dataset.shape[0], debug = args.verbose)

# 	end = time.time()
# 	print("ScalableSpectralDP:",end - start,"s")
# 	print("Saving ScalableSpectralDP...")
# 	sys.stdout.flush()

# 	# Now save it
# 	filename = os.path.splitext(args.data)[0]+'SpectralDP-'+str(M)+'.pickle'
# 	with open(filename, 'wb') as handle: 
# 		pickle.dump(algo, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 	print("ScalableSpectralDP saved.")
# 	sys.stdout.flush()
