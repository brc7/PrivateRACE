from race.hashes import *
import numpy as np 
import argparse
import sys
import os


''' Tool to evaluate ground-truth KDEs

'''

parser = argparse.ArgumentParser(description = "Evaluate ground truth KDEs. Produces a results file data.gtruth")
parser.add_argument("data", help="npy file with (n x d) data entries")
parser.add_argument("queries",help="npy file with (m x d) queries")
parser.add_argument("kernel_id", type=int, help="0: L2 LSH kernel, 1: Angular kernel")
parser.add_argument("bandwidth", type=float, help="density estimate bandwidth")
args = parser.parse_args()

# x = lambda a, b, c : a + b + c
# Gaussian kernel: 
# kernel = lambda x,y,w : np.exp(-1.0/(2*w) * np.linalg.norm(x - y)**2)
if args.kernel_id == 0: 
	kernel = lambda x,y,w : P_L2(np.linalg.norm(x-y),w)
elif args.kernel_id == 1:
	kernel = lambda x,y,w : P_SRP(x,y)**(int(w))
else: 
	print("Unsupported kernel id.") 
	sys.exit()

dataset = np.load(args.data)
queries = np.load(args.queries)

NQ,d = queries.shape
N,d = dataset.shape

print("Processing ground truth for",NQ," queries and",N," dataset vectors")
sys.stdout.flush()

results = np.zeros(NQ)

for j,data in enumerate(dataset):
	for i,query in enumerate(queries):
		results[i] += kernel(data,query,args.bandwidth)

	if j % 100 == 0:
		sys.stdout.write('\r')
		sys.stdout.write('Progress: {0:.4f}'.format(j/N * 100)+' %')
		sys.stdout.flush()

output_filename = os.path.splitext(args.data)[0]+'.gtruth'
np.savetxt(output_filename, results, delimiter=',')


