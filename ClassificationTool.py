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

parser = argparse.ArgumentParser(description = "Classifier Evaluation Tool - evaluate different sorts of RACE classifiers from private KDE sketches")
parser.add_argument("queries", help=".npy file with (n x d+1) data entries. First entry must be a class label.")
parser.add_argument("kernel_id", type=int, help="0: L2 LSH kernel, 1: Angular kernel")
parser.add_argument("bandwidth", type=float, help="Density estimate bandwidth")
parser.add_argument("epsilon", type=float, nargs ='+', help="Values of epsilon")
parser.add_argument("-r","--race", action='append', nargs = 2, help="RACE summary. Takes two args: (string filename, int class label). All summaries must be constructed with the same number of reps and same LSH function params / seed")
# parser.add_argument("-mle","--race", action='append', nargs = 2, help="RACE summary. Takes two args: (string filename, int class label)")

############################################################
# Check input args
args = parser.parse_args()
if args.race is None: 
	print("Please specify a RACE sketch for the classifier")
	sys.exit()
if len(args.race) < 2:
	print("Please specify more than one RACE sketch for the classifier")

############################################################
# Load queries 

queries = np.load(args.queries)
NQ,d = queries.shape
d = d - 1

############################################################
# Load RACEs

RACEs = []

for f,label in args.race: 
	handle = open(f,'rb')
	algo = pickle.load(handle)
	reps = algo.R

	############################################################
	# Construct LSH 
	if args.kernel_id == 0:
		np.random.seed(42)
		lsh = L2LSH(reps,d,args.bandwidth)
	elif args.kernel_id == 1:
		np.random.seed(42)
		lsh = SRPMulti(reps,d,int(args.bandwidth))
	else: 
		print("Unsupported kernel (hash function) id.") 
		sys.exit()

	RACEs.append((algo,lsh,label))

results = [] # all epsilon values

for j,ep in enumerate(args.epsilon): # for each epsilon
	print("Epsilon =",ep)
	for algo, lsh, label in RACEs:
		algo.set_epsilon(ep) # private wth this epsilon

	n_correct = 0
	for i,q in enumerate(queries): 
		true_label = int(q[0])
		query = q[1:]
		best_score = -1*np.inf
		best_label = -1
		for algo, lsh, label in RACEs:
			score = algo.query(lsh.hash(query))
			print('\t',label,':',score)
			if score > best_score:
				best_score = score
				best_label = int(label)
		print(true_label,':',best_label)
		if best_label == true_label: 
			n_correct += 1
		if i%1000 == 1:
			print('Accuracy (so far): {0:.4f}'.format(n_correct/i * 100)+' %')
			sys.stdout.flush()
	accuracy = n_correct/NQ
	print('Accuracy for epsilon = ',ep,' : ',accuracy)
	results.append((ep,accuracy))
