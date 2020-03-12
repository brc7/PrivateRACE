import numpy as np 
import argparse
import sys
import os

''' Tool to convert dataset to NPY format
'''

parser = argparse.ArgumentParser(description = "Convert dataset to NPY")
parser.add_argument("data", help=".csv file with (n x d) data entries")
parser.add_argument("-s","--skip", type=int, help="Skip first N rows of data file")
parser.add_argument("-cl","--dropcoll", type=int, help="Drop the first N (leading) cols of data file")
parser.add_argument("-ct","--dropcolt", type=int, help="Drop the last N (tail) cols of data file")
parser.add_argument("-d","--delimiter", help="Specify custom delimiter (default: whitespace)")
args = parser.parse_args()

skip = 0 
if args.skip: 
	skip = args.skip
dcl = 0
dct = 0
if args.dropcoll: 
	dcl = args.dropcoll
if args.dropcolt: 
	dct = args.dropcolt

output_filename = os.path.splitext(args.data)[0]+'.npy'

if args.delimiter: 
	dataset = np.loadtxt(args.data,delimiter = args.delimiter,skiprows = skip)
	np.save(output_filename, dataset[:,dcl:-dct])
else: 
	dataset = np.loadtxt(args.data,skiprows = skip)
	np.save(output_filename, dataset[:,dcl:-dct])


