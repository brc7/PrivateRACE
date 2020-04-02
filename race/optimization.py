import numpy as np 
from sklearn import linear_model
import sys 
import copy

def gradapprox(sketch, lsh, x, sigma, n_components):
	d = x.shape[0]
	directions = np.random.normal(size = (n_components,d))
	out = np.zeros_like(x)
	hc = lsh.hash(x)
	fx = sketch.query(hc)
	for di in directions:
		hci =  lsh.hash(x + sigma*di)
		out += (1.0 / (n_components * sigma)) * ( sketch.query(hci) - fx) * di
	return out

def gradapprox_dual(sketch, lsh, x, sigma, n_components):
	# Puts the PRP step into the gradient approximation
	d = x.shape[0]
	directions = np.random.normal(size = (n_components,d))
	out = np.zeros_like(x)
	hc_left = lsh.hash(x)
	hc_right = lsh.hash(-x)
	fx = sketch.query(hc_left)+sketch.query(hc_right)
	for di in directions:
		hci_left =  lsh.hash(x + sigma*di)
		hci_right =  lsh.hash(-x - sigma*di)
		out += (1.0 / (n_components * sigma)) * ( sketch.query(hci_left) + sketch.query(hci_right) - fx) * di
	return out


def greedyupdate_pstable(sketch, lsh, x):
	# greedy update procedure for p-stable lsh functions
	code = lsh.hash(x) # h is the (reps x 1) vector of hash codes
	# select a random hyperplane to "fix"
	i = np.random.randint(lsh.N) # i selects a random hash from h
	# find the largest L2Hash bucket in this RACE repetition
	optimal_code = np.argmax(sketch.counts[i,:])
	# if code were equal to optimal_code, then it would 
	# maximize the RACE score in this location. 
	# Let's update x so that x produces the optimal code
	if code[i] != optimal_code: 
		# move x the shortest distance necessary to land it in the center of this hash bucket
		D = (optimal_code + 0.5) * lsh.r - lsh.b[i]
		eta = (D - np.dot(x,lsh.W[i,:]))/np.sum(lsh.W[i,:]**2)
		# the locally-optimal update (the one that will move the hash code to the optimal one) is 
		# x = x + eta*lsh.W[i,:]
		# but we return eta*lsh.W[i,:] to allow the user to decide on different step sizes
		return eta*lsh.W[i,:]
	else: 
		return np.zeros_like(x)

def greedyupdate_srpmulti(sketch, lsh, x): 
	# much faster, but works only when range = 2**p
	i = np.random.randint(lsh.N_codes)
	p = lsh.p
	code = lsh.hash(x)[i]
	code = int(code)
	best_plane = None # the hyperplane that, when flipped, results in the largest improvement
	# best_plane stays at None if we don't find any 1-edit distance planes that are good
	best_count = sketch.counts[i,code]

	code_so_far = copy.deepcopy(code)
	for j in reversed(range(0, p+1)):
		c = 2**j # decimal-place value of bit position j
		if code_so_far - c < 0: 
			# then code (so far) has a zero in bit position j
			# and we want to try test_code = code + c
			test_code = code + c
		else: 
			# then code (so far) has a 1 in bit position j
			# and we want to try test_code = code - c
			test_code = code - c
			code_so_far = code_so_far - c # necessary for the if statement 

		if sketch.counts[i,test_code] > best_count: 
			best_count = sketch.counts[i,test_code]
			best_plane = j # best plane to flip is plane j 
	# optimal_code = np.argmax(sketch.counts[i,:])
	if best_plane is None: 
		return np.zeros_like(x)
	else: 
		hp = lsh.W[i*p + best_plane-1]
		return (np.dot(hp,x))/(np.linalg.norm(hp)**2)*hp


def gupta_method(sketch, lsh, x):
	d = x.shape[0]
	p = lsh.p
	h = np.sign( np.dot(lsh.W,x) )
	h = np.clip( h, 0, 1)
	h = np.reshape(h,(-1,p))
	i = np.random.randint(h.shape[0])
	nearh = np.zeros((p+1,p))
	nearsk = np.zeros(p+1)
	powersOfTwo = np.array([2**t for t in range(p)])

	# flip bit each time
	nearsk[0] = sketch.counts[i, int(np.dot(h[i], powersOfTwo))]
	nearh[0,:] = h[i]
	for j in range(1, p+1):
		key = h[i]
		# key[j] = 1- key[j]
		key[j-1] = 1- key[j-1]
		nearsk[j] = sketch.counts[i, int(np.dot(key, powersOfTwo))]
		nearh[j, :] = key

	# find max
	J = np.argmax(nearsk)
	if J ==0:
		return np.zeros_like(x)
	else:
		hp = lsh.W[i*p + J-1]
		# jump
		return (np.dot(hp,x))/(np.linalg.norm(hp)**2)*hp


# Convenience functions: 

def format_dataset(features,y,intercept = True): 
	N = features.shape[0]
	ones = np.ones((N,1))
	y = y.reshape((N,1))
	if intercept: 
		dataset = np.hstack((features,ones,y))
	else: 
		dataset = np.hstack((features,y))
	return dataset

def regression_loss(theta, data): 
	loss = 0 
	for e in data:
		loss += (e[-1] - np.dot(e[:-1], theta[:-1]))**2
	return loss


def optimal_linregress(x,y,intercept = True):
	reg = linear_model.LinearRegression()
	reg.fit(x, y)
	model = reg.coef_ 
	if intercept: 
		model = np.append(model, [reg.intercept_])
	return model

def construct_race_sketch(dataset, lsh, sketch, prp = True, verbose = True): 
	if verbose: 
		print("Sketching")
		sys.stdout.flush()
	for idx, x in enumerate(dataset): 
		sketch.add(lsh.hash(x))
		if prp: 
			sketch.add(lsh.hash(-x))
		if verbose: 
			if idx % 100 == 0: 
				print('',end = '.')
				sys.stdout.flush()
	sketch.set_epsilon(None)
	return sketch


def accelerated_race_zgd(sketch, lsh, n_iters, eta, beta, sigma, n_components, 
	verbose = True, loss = None, dual = False): 
	# sketch: race sketch
	# lsh: lsh used to construct sketch
	# loss: (optional) a function of one variable applied to the result of each iteration
	# for linear regression, could be regression_loss(theta) to give performance on training / validation set
	# for mode finding / other experiments, could be distance between theta and the true (known) mode
	# dual: use gradapprox_dual (i.e. put the PRP into the gradient descent step)
	theta = np.zeros(lsh.d,dtype = np.float64)
	theta[-1] = -1 # for y
	race_values = np.zeros(n_iters)
	losses = np.zeros(n_iters)

	if dual: 
		gradient = gradapprox_dual
	else: 
		gradient = gradapprox

	v = 0 
	for i in range(n_iters): 
		v = beta * v + (1 - beta) * gradient(sketch,lsh,theta,sigma,n_components)
		theta = theta - eta*v
		theta[-1] = -1 # project back onto the constraint
		race_values[i] = sketch.query(lsh.hash(theta))
		if loss is not None: 
			losses[i] = loss(theta) # previously, regression_loss(theta,dataset)
		if verbose: 
			if i % 10 == 0:
				# print('',end='.')
				print(np.linalg.norm(v),race_values[i],losses[i])
				sys.stdout.flush()
	return (theta, race_values, losses)


def race_zgd(sketch, lsh, n_iters, eta, sigma, n_components, 
	verbose = True, loss = None, dual = False):
	# sketch: race sketch
	# lsh: lsh used to construct sketch
	# loss: (optional) a function of one variable applied to the result of each iteration
	# for linear regression, could be regression_loss(theta) to give performance on training / validation set
	# for mode finding / other experiments, could be distance between theta and the true (known) mode
	# dual: use gradapprox_dual (i.e. put the PRP into the gradient descent step)
	theta = np.zeros(lsh.d,dtype = np.float64)
	theta[-1] = -1 # for y
	race_values = np.zeros(n_iters)
	losses = np.zeros(n_iters)
	
	if dual: 
		gradient = gradapprox_dual
	else: 
		gradient = gradapprox

	for i in range(n_iters):
		v = gradient(sketch,lsh,theta,sigma,n_components) 
		theta = theta - eta*v
		theta[-1] = -1 # project back onto the constraint
		race_values[i] = sketch.query(lsh.hash(theta))
		if loss is not None: 
			losses[i] = loss(theta) # previously, regression_loss(theta,dataset)
		if verbose: 
			if i % 10 == 0:
				# print('',end='.')
				print(np.linalg.norm(v),race_values[i],losses[i])
				sys.stdout.flush()
	return (theta, race_values, losses)



def race_greedy(sketch, lsh, n_iters, alpha,
	verbose = True, loss = None):
	# only works with PRP type sketches
	# typical value for alpha = 0.3
	# sketch: race sketch
	# lsh: lsh used to construct sketch
	# loss: (optional) a function of one variable applied to the result of each iteration
	# for linear regression, could be regression_loss(theta) to give performance on training / validation set
	# for mode finding / other experiments, could be distance between theta and the true (known) mode

	theta = np.zeros(lsh.d,dtype = np.float64)
	theta[-1] = -1 # for y
	race_values = np.zeros(n_iters)
	losses = np.zeros(n_iters)

	for i in range(n_iters):
		v = greedyupdate_srpmulti(sketch, lsh, theta)
		# v = gupta_method(sketch, lsh, theta)
		theta = theta - (1+alpha)*v
		theta[-1] = -1 # project back onto the constraint
		race_values[i] = sketch.query(lsh.hash(theta))
		if loss is not None: 
			losses[i] = loss(theta) # previously, regression_loss(theta,dataset)
		if verbose: 
			if i % 10 == 0:
				# print('',end='.')
				print(np.linalg.norm(v),race_values[i],losses[i])
				sys.stdout.flush()
	return (theta, race_values, losses)

