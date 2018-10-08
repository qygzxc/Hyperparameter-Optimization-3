# _*_ coding:utf-8 _*_

'''
Author: Ruan Yang
Email: ruanyang_njut@163.com

Purpose: learning Hyperparameter optimization methods

1. Grid serach (sklearn)
2. Random search (HyperOpt or sklearn)
3. Bayesian search (HyperOpt)

References: https://github.com/hyperopt/hyperopt
            https://github.com/WillKoehrsen/hyperparameter-optimization
            
Bayesian Model-Based optimization contain four parts
1. Objective: what we want to minimize
2. Domain: values of the parameters over which to minimize the objective
3. Hyperparameter optimization function: how the surrogate function is \
   built and the next values are proposed 
4. Trials consisting of score, parameters pairs         
'''

# import library

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# defiend the objective function
# There used a simple polynomial function
# This function has one global minimum over the range we define it as \
# well as one local minimum.

def objective(x):
	'''
	Objective function to minimize
	'''
	# create the polynomial object
	f=np.poly1d([1,-2,-28,28,12,-26,100])
	
	# Return the value of the polynomial
	return f(x)*0.05
	
# space over which to evluate the function is (-5,6)

x=np.linspace(-5,6,10000)
y=objective(x)

miny=np.min(y)

# get the miny corresponding x value
minx=x[np.argmin(y)]

# visualize the function

fig,axis=plt.subplots(figsize=(8,6))
plt.style.use("fivethirtyeight")
axis.set_title("Objective Function")
axis.set_xlabel("x")
axis.set_ylabel("y")
axis.grid(True)
axis.vlines(minx,np.min(y)-50,np.max(y),linestyles = '--', colors = 'r')
plt.plot(x,y)
plt.savefig("figure-1.jpg",dpi=300)
plt.show()

# Domain
# The domain is the values of x over which we evaluate the function
# First we can use a uniform distribution over the space our function is defined.
#import hyperopt library

from hyperopt import hp

# create the domain space

space=hp.uniform('x',-5,6)

# visualize the uniform samples in the x zone

from hyperopt.pyll.stochastic import sample

samples=[]

# Sample 10000 values from the range

for _ in range(10000):
	samples.append(sample(space))
	
# Histogram of the values

fig,axis=plt.subplots(figsize=(8,6))
plt.style.use("fivethirtyeight")
axis.set_title("Domain space")
axis.set_xlabel("x")
axis.set_ylabel("Frequency")
axis.grid(True)
axis.hist(samples,bins=30,edgecolor='red')
plt.savefig("figure-2.jpg",dpi=300)
plt.show()

# Bayesian optimization methods 
# 1. initially at random as it explores the domain space
# 2. but then over time, it will "focus" on the most promising values

# Using suggest method defined in hyperopt

from hyperopt import rand,tpe

# create the algorithms

tpe_algo=tpe.suggest
rand_algo=rand.suggest

# History: Storing the history is as simple as making a Trials object \
# that we pass into the function call.

from hyperopt import Trials

# create two trails objects

tpe_trials=Trials()
rand_trials=Trials

# run the optimization
# using fmin method

from hyperopt import fmin

# Run 2000 evals with the algorithm
# fn: objective function
# space: defined the x value sample space
# algo: the search method
# trials: 
# max_evals: maximum number of iterations
# rstate: for reproducible results across multiple runs

tpe_best=fmin(fn=objective,space=space,algo=tpe_algo,trials=tpe_trials,\
max_evals=2000,rstate=np.random.RandomState(50))

print("#---------------------------------#")
print(tpe_best)
print("#---------------------------------#")
print("\n")

#rand_best=fmin(fn=objective,space=space,algo=rand_algo,trials=rand_trials,max_evals=2000,rstate=np.random.RandomState(50))

rand_best=fmin(fn=objective,space=space,algo=rand_algo,trials=rand_trials,\
max_evals=2000,rstate=np.random.RandomState(50))

print("#---------------------------------#")
print(rand_best)
print("#---------------------------------#")
print("\n")

print("#---------------------------------#")
print('Minimum loss attained with TPE:    {:.4f}'.format(\
tpe_trials.best_trial['result']['loss']))
print('Minimum loss attained with random: {:.4f}'.format(\
rand_trials.best_trial['result']['loss']))
print('Actual minimum of f(x):            {:.4f}'.format(miny))
print("#---------------------------------#")
print("\n")

print("#---------------------------------#")
print('Number of trials needed to attain minimum with TPE:    \
{}'.format(tpe_trials.best_trial['misc']['idxs']['x'][0]))
print('Number of trials needed to attain minimum with random: \
{}'.format(rand_trials.best_trial['misc']['idxs']['x'][0]))
print("#---------------------------------#")
print("\n")

print("#---------------------------------#")
print('Best value of x from TPE:      {:.4f}'.format(tpe_best['x']))
print('Best value of x from random:   {:.4f}'.format(rand_best['x']))
print('Actual best value of x:        {:.4f}'.format(minx))
print("#---------------------------------#")
print("\n")

# Results
# take a look at the trials objects

tpe_results=pd.DataFrame({"loss":[x['loss'] for x in tpe_trials.results],\
"iteration":tpe_trials.idxs_vals[0]['x'],"x":tpe_trials.idxs_vals[1]['x']})

print("#---------------------------------#")
print(tpe_results.head())
print("#---------------------------------#")
print("\n")

tpe_results['rolling_average_x'] = tpe_results['x'].rolling(50).mean().\
fillna(method = 'bfill')
tpe_results['rolling_average_loss'] = tpe_results['loss'].rolling(50).\
mean().fillna(method = 'bfill')

print("#---------------------------------#")
print(tpe_results.head())
print("#---------------------------------#")
print("\n")

fig,axis=plt.subplots(figsize=(8,6))
plt.style.use("fivethirtyeight")
axis.set_title("TPE Sequence of Values",size=24)
axis.set_xlabel("Iteration",size=22)
axis.set_ylabel("x value",size=22)
axis.hlines(minx,0,2000,linestyles='--',colors='r')
axis.grid(True)
axis.plot(tpe_results['iteration'],tpe_results['x'],'bo',alpha=0.5)
plt.savefig("figure-3.jpg",dpi=300)
plt.show()

fig,axis=plt.subplots(figsize=(8,6))
plt.style.use("fivethirtyeight")
axis.hist(tpe_results["x"],bins=50,edgecolor="red")
axis.title("Histogram of TPE Values")
axis.xlabel("Value of x")
axis.ylabel("Count")
plt.savefig("figure-4.jpg",dpi=300)
plt.show()

# sort with best loss first

tpe_results = tpe_results.sort_values('loss', ascending = True).reset_index()

fig,axis=plt.subplots(figsize=(8,6))
plt.style.use("fivethirtyeight")
axis.plot(tpe_results['iteration'],tpe_results['loss'],'bo',alpha=0.3)
axis.xlabel("Iteration")
axis.ylabel("loss")
axis.title("TPE Sequence of Losses")
plt.savefig("figure-5.jpg",dpi=300)
plt.show()
print('Best Loss of {:.4f} occured at iteration {}'.format(tpe_results['loss'][0],\
tpe_results['iteration'][0]))

# Random result

rand_results=pd.DataFrame({'loss':[x['loss'] for x in rand_trials.results],\
'iteration': rand_trials.idxs_vals[0]['x'],'x': rand_trials.idxs_vals[1]['x']})

print("#---------------------------------#")
print(rand_results.head())
print("#---------------------------------#")
print("\n")

fig,axis=plt.subplots(figsize=(8,6))
plt.style.use("fivethirtyeight")
axis.set_title("Random Sequence of Values",size=24)
axis.set_xlabel("Iteration",size=22)
axis.set_ylabel("x value",size=22)
axis.hlines(minx,0,2000,linestyles='--',colors='r')
axis.grid(True)
axis.plot(tpe_results['iteration'],tpe_results['x'],'bo',alpha=0.5)
plt.savefig("figure-6.jpg",dpi=300)
plt.show()

# sort with best loss first

rand_results=rand_results.sort_values('loss',ascending = True).reset_index()

fig,axis=plt.subplots(figsize=(8,6))
plt.style.use("fivethirtyeight")
axis.hist(tpe_results["x"],bins=50,edgecolor="red")
axis.title("Histogram of Random Values")
axis.xlabel("Value of x")
axis.ylabel("Count")
plt.savefig("figure-7.jpg",dpi=300)
plt.show()

print('Best Loss of {:.4f} occured at iteration {}'.\
format(rand_results['loss'][0], rand_results['iteration'][0]))

# Better domain space
# Using graph,experience and knowledge to inform our choice of a domain space

# Normally distributed space
space = hp.normal('x', 4.9, 0.5)

samples = []

# Sample 10000 values from the range
for _ in range(10000):
    samples.append(sample(space))
    

# Histogram of the values
fig,axis=plt.subplots(figsize=(8,6))
plt.style.use("fivethirtyeight")
axis.hist(samples,bins = 20,edgecolor = 'black')
axis.xlabel('x')
axis.ylabel('Frequency')
axis.title('Domain Space')
plt.savefig("figure-8.jpg",dpi=300)
plt.show()

# more useful trails object

from hyperopt import STATUS_OK
from timeit import default_timer as timer

def objective(x):
	'''
	Objective function to minimize with smarter return values
	'''
	
	# Create the polynomial object
	f=np.poly1d([1,-2,-28,28,12,-26,100])
	
	# evaluate the function
	start=timer()
	loss=f(x)*0.05
	end=timer()
	
	# calculate time to evaluate
	time_elapsed=end-start
	
	results={"loss":loss,"status":STATUS_OK,"x":x,"time":time_elapsed}
	
	# return dictionary
	return results
	
# new trials object

trials=Trials()

# run 2000 evals with the tpe algorithm
best = fmin(fn=objective,space=space,algo=tpe_algo,trials=trials,\
max_evals=2000,rstate= np.random.RandomState(120))

results=trials.results
results[:2]

# results into a dataframe

results_df=pd.DataFrame({"time":[x["time"] for x in results],"loss":\
[x["loss"] for x in results],'x':[x['x'] for x in results],\
"iteration": list(range(len(results)))})

# sort with lowest loss on top
results_df=results_df.sort_values("loss",ascending=True)
results_df.head()

fig,axis=plt.subplots(figsize=(8,6))
plt.style.use("fivethirtyeight")
axis.hist(results_df['x'],bins=50,edgecolor='k')
axis.xlabel('Value of x')
axis.ylabel('Count')
axis.title('Histogram of TPE Values')
plt.savefig("figure-9.jpg",dpi=300)
plt.show()

fig,axis=plt.subplots(figsize=(8,6))
sns.kdeplot(results_df['x'],label='Normal Domain')
sns.kdeplot(tpe_results['x'],label='Uniform Domain')
axis.legend()
axis.xlabel('Value of x')
axis.ylabel('Density')
axis.title('Comparison of Domain Choice using TPE');
plt.savefig("figure-10.jpg",dpi=300)
plt.show()

print('Lowest Value of the Objective Function = {:.4f} at x = {:.4f} \
found in {:.0f} iterations.'.format(results_df['loss'].min(),\
results_df.loc[results_df['loss'].idxmin()]['x'],\
results_df.loc[results_df['loss'].idxmin()]['iteration']))


# One-line Optimization
# Just because you can do it in one line doesn't mean you should! 

best=fmin(fn=lambda x:np.poly1d([1,-2,-28,28,12,-26,100])(x)*0.05,\
space=hp.normal('x',4.9,0.5),algo=tpe.suggest,max_evals = 2000)
