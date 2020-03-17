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

parser = argparse.ArgumentParser(description = "Density Experiment Evaluation (DEV) tool - evaluate private function summaries")
parser.add_argument("queries", help=".npy file with (n x d) data entries")
parser.add_argument("gtruth", help=".gtruth file with n ground truth KDE values")
parser.add_argument("epsilon", type=float, nargs ='+', help="values of epsilon")
parser.add_argument("-r","--race", nargs = 3, help="RACE summary (filename, int kernel_id, float bandwidth)")
parser.add_argument("-b","--bernstein", nargs = 2, help="Bernstein summary (filename, int scale factor)")
parser.add_argument("-kme","--kmerelease", nargs = 3, help="KME summary (filename, int kernel_id, float bandwidth)")

# parser.add_argument("-s","--spectral", type=int, help="Prepare SpectralDP with a K-lattice")
args = parser.parse_args()

queries = np.load(args.queries)
NQ,d = queries.shape
gtruth = np.loadtxt(args.gtruth,delimiter = '\n')

if args.race: 
	print("Querying RACE")
	sys.stdout.flush()


	f = open(args.race[0],'rb')
	kernel_id = int(args.race[1])
	bandwidth = float(args.race[2])
	algo = pickle.load(f)
	reps = algo.R

	if kernel_id == 0:
		np.random.seed(42)
		lsh = L2LSH(reps,d,bandwidth)
	elif kernel_id == 1:
		np.random.seed(42)
		lsh = SRPMulti(reps,d,int(bandwidth))
	else: 
		print("Unsupported kernel (hash function) id.") 
		sys.exit()

	start = time.time()
	results = [] # all epsilon values
	values = np.zeros_like(gtruth) # results for each query

	for ep in args.epsilon: # for each epsilon
		algo.set_epsilon(ep) # private wth this epsilon
		for i,q in enumerate(queries): # for each query
			val = algo.query(lsh.hash(np.array(q)))
			values[i] = val
			if i%100 == 0: 
				sys.stdout.write('\r')
				sys.stdout.write('Epsilon: {0:.4f} Progress: {0:.4f}'.format(ep,i/NQ * 100)+' %')
				sys.stdout.flush()
		sys.stdout.write('\n')

		err = np.abs(val - gtruth) # error vector
		results.append((np.mean(err),np.std(err))) # mean,std error 
	end = time.time()
	print("Query time: (avg, ms) ",(end-start)*1000/(gtruth.shape[0]*NQ))
	print(results)

if args.bernstein: 
	print("Querying Bernstein")
	sys.stdout.flush()

	f = open(args.bernstein[0],'rb')
	scale_factor = int(args.bernstein[1])
	algo = pickle.load(f)
	start = time.time()
	results = []
	values = np.zeros_like(gtruth)
	for ep in args.epsilon: 
		algo.set_epsilon(ep)
		for i,q in enumerate(queries):
			val = algo.query(q/scale_factor)
			values[i] = val
			if i%100 == 0: 
				sys.stdout.write('\r')
				sys.stdout.write('Epsilon: {0:.4f} Progress: {0:.4f}'.format(ep,i/NQ * 100)+' %')
				sys.stdout.flush()
		sys.stdout.write('\n')
		err = np.abs(val - gtruth/args.N)
		results.append((np.mean(err),np.std(err)))
	end = time.time()
	print("Query time: (avg, ms) ",(end-start)*1000/(gtruth.shape[0]*NQ))
	print(results)

if args.kmerelease: 
	print("Querying KME Release")
	sys.stdout.flush() 

	f = open(args.kmerelease[0],'rb')
	kernel_id = int(args.kmerelease[1])
	bandwidth = float(args.kmerelease[2])
	if kernel_id == 0: 
		kernel = lambda x,y : P_L2(np.linalg.norm(x-y),bandwidth)
	elif kernel_id == 1:
		kernel = lambda x,y : P_SRP(x,y)**(int(bandwidth))
	else: 
		print("Unsupported kernel id.") 
		sys.exit()

	algo = pickle.load(f)
	start = time.time()
	results = []
	values = np.zeros_like(gtruth)
	for ep in args.epsilon: 
		algo.set_epsilon(ep)
		for i,q in enumerate(queries):
			val = algo.query(q,kernel)
			values[i] = val
			if i%100 == 0: 
				sys.stdout.write('\r')
				sys.stdout.write('Epsilon: {0:.4f} Progress: {0:.4f}'.format(ep,i/NQ * 100)+' %')
				sys.stdout.flush()
		sys.stdout.write('\n')

		err = np.abs(val - gtruth/args.N)
		results.append((np.mean(err),np.std(err)))
	end = time.time()
	print("Query time: (avg, ms) ",(end-start)*1000/(gtruth.shape[0]*NQ))
	print(results)






