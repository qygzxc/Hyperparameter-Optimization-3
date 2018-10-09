# _*_ coding:utf-8 _*_

'''
Author: Ruan Yang
Email: ruanyang_njut@163.com

Purpose: learning Hyperparameter optimization methods
         optimize the hyperparameters of a Gradient Boosting Machine \
         using the Hyperopt library 
         
Gradient Boosting Machine (GBM): lightgbm library

1. Grid serach (sklearn)
2. Random search (HyperOpt or sklearn)
3. Bayesian search (HyperOpt)

References: https://github.com/hyperopt/hyperopt
            https://github.com/WillKoehrsen/hyperparameter-optimization
            
Attention: we also have to remember that the hyperparameters are optimized 
           on the validation data.
'''

# pandas and numpy for data manipulation

import pandas as pd
import numpy as np

# GBM
import lightgbm as lgb

# Evaluating of the model

from sklearn.model_selection import KFold

#MAX_EVALS=500
MAX_EVALS=5
N_FOLDS=10

# Data: using kaggle data
# Task: supervised machine learning classification task:
# iven past data, we want to train a model to predict a binary outcome \
# on testing data

# read in data and separate into training and testing sets

data=pd.read_csv("caravan-insurance-challenge.csv")

train=data[data['ORIGIN']=='train']
test=data[data['ORIGIN']=="test"]

# Extract the labels and format properly

train_labels=np.array(train['CARAVAN'].astype(np.int32)).reshape((-1,))
test_labels=np.array(test['CARAVAN'].astype(np.int32)).reshape((-1,))

# Drop the unneeded columns
# remove identification and labels

train=train.drop(columns = ['ORIGIN', 'CARAVAN'])
test=test.drop(columns = ['ORIGIN', 'CARAVAN'])

# Convert to numpy array for splitting in cross validation
# CV=cross validation

features=np.array(train)
test_features=np.array(test)
labels=train_labels[:]

print("#------------------------------#")
print("Train shape: ",train.shape)
print("Test shape: ",test.shape)
print(train.head())
print("#------------------------------#")
print("\n")

# Distribution of Label

import matplotlib.pyplot as plt
import seaborn as sns

fig,axis=plt.subplots(figsize=(8,6))
#plt.style.use("fivethirtyeight")
axis.set_title("Counts of labels")
axis.set_xlabel("Label")
axis.set_ylabel("Count")
axis.grid(True)
axis.hist(labels,edgecolor='red')
plt.savefig("figure-1.jpg",dpi=300)
plt.show()

# Based on labels distribution confirm the metrics
# common classification metric of Receiver Operating Characteristic Area \
# Under the Curve (ROC AUC)
# Randomly guessing on a classification problem will yield an ROC AUC of 0.5
# a perfect classifier has an ROC AUC of 1.0

# GBM with default hyperparameters

# model with default hyperparameters

model=lgb.LGBMClassifier()

print("#------------------------------#")
print(model)
print("#------------------------------#")
print("\n")

from sklearn.metrics import roc_auc_score
from timeit import default_timer as timer

start=timer()

# fit train data

model.fit(features,labels)

# count the fit model time

train_time=timer()-start

# predict probability

predictions=model.predict_proba(test_features)[:,1]
auc=roc_auc_score(test_labels,predictions)

print("#------------------------------#")
print('The baseline score on the test set is {:.4f}.'.format(auc))
print('The baseline training time is {:.4f} seconds'.format(train_time))
print("#------------------------------#")
print("\n")

# Random Search
# 1. defined the search space
# 2. random search is very effective, returning nearly as good results \
#    as grid search with a significant reduction in time spent searching
# Random search have four parts like Bayesian hyperparameter optimization

# 1. Domain: values over which to search
# 2. Optimization algorithm: pick the next values at random! (yes this \
#    qualifies as an algorithm)
# 3. Objective function to minimize: in this case our metric is \
#    cross validation ROC AUC
# 4. Results history that tracks the hyperparameters tried and the \
#    cross validation metric

# Random search and Grid Search implemented in sklearn
# In the search procedure using Early Stopping

import random

# Domain for random search

# 1. Random search and Bayesian optimization both search for \
#    hyperparameters from a domain
# 2. Random (or grid search) this domain is called a hyperparameter \
#    grid and uses discrete values for the hyperparameters

# list the hyperparameters in lightgb

print("#------------------------------#")
print(lgb.LGBMClassifier())
print("#------------------------------#")
print("\n")

# Hyperparameter grid

param_grid={"class_weight":[None,"balanced"],\
"boosting_type":["gbdt","goss","dart"],\
"num_leaves":list(range(30,150)),\
"learning_rate":list(np.logspace(np.log(0.005),np.log(0.2),\
base = np.exp(1),num = 1000)),\
"subsample_for_bin":list(range(20000,300000,20000)),\
"min_child_samples":list(range(20,500,5)),\
'reg_alpha': list(np.linspace(0, 1)),\
'reg_lambda': list(np.linspace(0, 1)),\
'colsample_bytree': list(np.linspace(0.6, 1, 10))}

# Subsampling (only applicable with "goss")

subsample_dist=list(np.linspace(0.5,1,100))

# Take a look about learning_rate distribution

fig,axis=plt.subplots(figsize=(8,6))
#plt.style.use("fivethirtyeight")
axis.set_title("Learning rate distribution")
axis.set_xlabel("Learning rate")
axis.set_ylabel("Count")
axis.grid(True)
axis.hist(param_grid["learning_rate"],color="r",edgecolor='red')
plt.savefig("figure-2.jpg",dpi=300)
plt.show()

fig,axis=plt.subplots(figsize=(8,6))
#plt.style.use("fivethirtyeight")
axis.set_title("Number of Leaves distribution")
axis.set_xlabel("Learning Number of Leaves")
axis.set_ylabel("Count")
axis.grid(True)
axis.hist(param_grid["num_leaves"],color="m",edgecolor='red')
plt.savefig("figure-3.jpg",dpi=300)
plt.show()

# sampling from hyperparameter domain
# randomly sample parameters for gbm

params={key:random.sample(value,1)[0] for key,value in param_grid.items()}

print("#------------------------------#")
print(params)
print("#------------------------------#")
print("\n")

# To add a subsample ratio if the boosting_type is not goss, we can use \
# an if statement

params["subsample"]=random.sample(subsample_dist,1)[0] if params["boosting_type"] !=\
"goss" else 1.0

print("#------------------------------#")
print(params)
print("#------------------------------#")
print("\n")

# Cross validation with early stopping in LightGBM
# 1. scikit-learn cross validation api does not include the option for \
#    early stopping
# 2. the LightGBM cross validation function with 100 early stopping rounds

# create a lgb dataset

train_set=lgb.Dataset(features,label=labels)

# cv funtion embedding in lightgbm need these parameters
# 1. the training data
# 2. the number of training rounds
# 3. the number of folds
# 4. the metric
# 5. the number of early stopping rounds
# 6. other arguments

# Early stopping purpose:early stopping to stop training when the \
#   validation score has not improved for 100 estimators

# perform cross validation with 10 folds

r=lgb.cv(params,train_set,num_boost_round=10000,nfold=10,metrics="auc",\
early_stopping_rounds=100,verbose_eval=False,seed=50)

# Highest score

r_best=np.max(r["auc-mean"])

# standard deviation of best score

r_best_std=r["auc-stdv"][np.argmax(r["auc-mean"])]

print("#------------------------------#")
print('The maximium ROC AUC on the validation set was {:.5f} with std \
of {:.5f}.'.format(r_best, r_best_std))
print('The ideal number of iterations was {}.'.format(np.argmax(r['auc-mean']) + 1))
print("#------------------------------#")
print("\n")

# Dataframe to hold cv results

random_results=pd.DataFrame(columns=["loss","params","iteration","estimators",\
"time"],index=list(range(MAX_EVALS)))

# train set, validation set and test set
# 1. train set: train ML model
# 2. validation set: tune the model
# 3. test set: predication model

# Build validation model using KFold
# A better approach than drawing the validation set from the training \
# data (thereby limiting the amount of training data we have) is KFold \
# cross validation.

def random_objective(params,iteration,n_folds=N_FOLDS):
	'''
	Random search objective function.
	Takes in hyperparameters and returns a list of results to be saved
	'''
	start=timer()
	
	# perform n_folds corss validation
	cv_results=lgb.cv(params,train_set,num_boost_round=10000,nfold=n_folds,\
	early_stopping_rounds=100,metrics='auc',seed=50)
	
	end=timer()
	
	best_score=np.max(cv_results["auc-mean"])
	
	# loss must be minimized
	
	loss=1-best_score
	
	# boosting rounds that returned the highest cv score
	n_estimators=int(np.argmax(cv_results["auc-mean"])+1)
	
	# return list of results
	
	return [loss,params,iteration,n_estimators,end-start]

# Random Search Implementation

random.seed(50)

# Iterate through the specified number of evaluations

for i in range(MAX_EVALS):
	# randomly sample parameters for gbm
	params={key:random.sample(value,1)[0] for key,value in param_grid.items()}
	
	print(params)
	
	if params["boosting_type"]=="goss":
		# cannot subsample with goss
		params["subsample"]=1.0
	else:
		# subsample supported for gbdt and dart
		params["subsample"]=random.sample(subsample_dist,1)[0]
		
	results_list=random_objective(params,i)	
	
	# add results to next row in dataframe
	random_results.loc[i,:]=results_list
	
# sort results by best validation score
# That's to say the first value get the best score

random_results.sort_values("loss",ascending=True,inplace=True)
random_results.reset_index(inplace=True,drop=True)
random_results.head()

print("#------------------------------#")
print(random_results.loc[0,"params"])
print("#------------------------------#")
print("\n")

# Find the best parameters and number of estimators

best_random_params=random_results.loc[0,"params"].copy()
best_random_estimators=int(random_results.loc[0,"estimators"])
best_random_model=lgb.LGBMClassifier(n_estimators=best_random_estimators,\
n_jobs=1,objective="binary",**best_random_params,random_state=50)

# Fit on the training data

best_random_model.fit(features,labels)

# make test predictions

predictions=best_random_model.predict_proba(test_features)[:,1]

print("#------------------------------#")
print('The best model from random search scores {:.4f} on the test data.\
'.format(roc_auc_score(test_labels, predictions)))
print("\n")
print('This was achieved using {} search iterations.'.format(random_results.loc[0, 'iteration']))
print("\n")
print("#------------------------------#")
print("\n")

# bayesian hyperparameter optimization using hyperopt

# 1. Objective function
# 2. Domain space
# 3. Hyperparameter optimization algorithm
# 4. History of results

# Objective function
# Difference: This objective function will still take in the hyperparameters \
# but it will return not a list but a dictionary.

import csv
from hyperopt import STATUS_OK
from timeit import default_timer as timer

def objective(params,n_folds=N_FOLDS):
	'''
	Objective function for Gradient Boosting Machine Hyperparameter Optimization
	'''
	
	# keep track of evals
	global ITERATION
	ITERATION += 1
	
	# retrieve the subsample if present otherwise set to 1.0
	subsample=params["boosting_type"].get("subsample",1.0)
	
	# extract the boosting type
	params["boosting_type"]=params["boosting_type"]["boosting_type"]
	params["subsample"]=subsample
	
	# make sure parameters that need to be integer are integers
	for parameter_name in ["num_leaves","subsample_for_bin","min_child_samples"]:
		params[parameter_name]=int(params[parameter_name])
		
	start=timer()
	
	# perform n_folds cross validation
	
	cv_results=lgb.cv(params,train_set,num_boost_round=10000,nfold=n_folds,\
	early_stopping_rounds=100,metrics="auc",seed=50)
	
	run_time=timer()-start
	
	# extract the best score
	
	best_score=np.max(cv_results["auc-mean"])
	
	# loss must be minimized
	
	loss=1-best_score
	
	# Boosting rounds that returned the highest cv score
	
	n_estimators=int(np.argmax(cv_results["auc-mean"])+1)
	
	# Write to the csv file ('a' means append)
	of_connection = open(out_file, 'a')
	writer = csv.writer(of_connection)
	writer.writerow([loss, params, ITERATION, n_estimators, run_time])
	
	# Dictionary with information for evaluation
	return {'loss': loss, 'params': params, 'iteration': ITERATION,\
	'estimators': n_estimators,'train_time': run_time, 'status': STATUS_OK}
	
# Domain space

from hyperopt import hp
from hyperopt.pyll.stochastic import sample

# create the learning rate

learning_rate={"learning_rate":hp.loguniform("learning_rate",np.log(0.05),\
np.log(0.2))}

learning_rate_dist=[]

# Draw 10000 samples from the learning rate domain

for _ in range(10000):
	learning_rate_dist.append(sample(learning_rate)["learning_rate"])

fig,axis=plt.subplots(figsize=(8,6))
#plt.style.use("fivethirtyeight")
axis.set_title("Learning Rate distribution")
axis.set_xlabel("Learning Rate")
axis.set_ylabel("Density")
axis.grid(True)
#axis.hist(param_grid["num_leaves"],color="m",edgecolor='red')
sns.kdeplot(learning_rate_dist,color="red",linewidth=2,shade=True)
plt.savefig("figure-4.jpg",dpi=300)
plt.show()

# Discrete uniform distribution

num_leaves={"num_leaves":hp.quniform("num_leaves",30,150,1)}
num_leaves_dist=[]

# sample 10000 times from the number of leaves distribution

for _ in range(10000):
	num_leaves_dist.append(sample(num_leaves)["num_leaves"])

fig,axis=plt.subplots(figsize=(8,6))
#plt.style.use("fivethirtyeight")
axis.set_title("Number of Leaves distribution")
axis.set_xlabel("Number of Leaves")
axis.set_ylabel("Density")
axis.grid(True)
#axis.hist(param_grid["num_leaves"],color="m",edgecolor='red')
sns.kdeplot(num_leaves_dist,color="green",linewidth=2,shade=True)
plt.savefig("figure-5.jpg",dpi=300)
plt.show()

# conditional domain

# boosting type domain

boosting_type = {'boosting_type': hp.choice('boosting_type',[{'boosting_type': \
'gbdt', 'subsample': hp.uniform('subsample', 0.5, 1)}, {'boosting_type': \
'dart', 'subsample': hp.uniform('subsample', 0.5, 1)},{'boosting_type': \
'goss', 'subsample': 1.0}])}

# Draw a sample

params=sample(boosting_type)

print("#------------------------------#")
print(params)
print("#------------------------------#")
print("\n")

# retrieve the subsample if present otherwise set to 1.0

subsample=params["boosting_type"].get("subsample",1.0)

# extract the boosting type

params['boosting_type'] = params['boosting_type']['boosting_type']
params['subsample'] = subsample

print("#------------------------------#")
print(params)
print("#------------------------------#")
print("\n")

# complete bayesian domain

# Define the search space
# main variables
# 1. class_weight
# 2. boosting_type
# 3. num_leaves
# 4. learning_rate
# 5. subsample_for_bin
# 6. min_child_samples
# 7. reg_alpha
# 8. reg_lambda
# 9. colsample_bytree

space={'class_weight': hp.choice('class_weight', [None, 'balanced']),\
'boosting_type': hp.choice('boosting_type', [{'boosting_type': 'gbdt',\
'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, \
{'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},\
{'boosting_type': 'goss', 'subsample': 1.0}]),\
'num_leaves': hp.quniform('num_leaves', 30, 150, 1),\
'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),\
'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),\
'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),\
'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),\
'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),\
'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)}

# example of sampling from the domain 

# sample from the full space

x=sample(space)

# conditional logic to assign top-level keys

subsample=x["boosting_type"].get("subsample",1.0)
x['boosting_type'] = x['boosting_type']['boosting_type']
x['subsample'] = subsample

print("#------------------------------#")
print(x)
print("#------------------------------#")
print("\n")

#subsample=x["boosting_type"].get("subsample",1.0)
#x['boosting_type'] = x['boosting_type']['boosting_type']
#x['subsample'] = subsample
#
#print("#------------------------------#")
#print(x)
#print("#------------------------------#")
#print("\n")

#optimization algorithm

from hyperopt import tpe

# optimization algorithm

tpe_algorithm=tpe.suggest

# results history
# Two methods used to record the results
# 1.A Trials object that stores the dictionary returned from the objective function
# 2.Writing to a csv file every iteration
#   The csv file option also lets us monitor the results of an on-going experiment.

from hyperopt import Trials

# keep track of results
bayes_trials=Trials()

# File to save first results

out_file="gbm_trials.csv"
of_connection=open(out_file,"w")
writer=csv.writer(of_connection)

# writes the headers of the file
writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
of_connection.close()

from hyperopt import fmin

# global variable

global ITERATION

ITERATION=0

# run optimization

best=fmin(fn=objective,space=space,algo=tpe.suggest,max_evals=MAX_EVALS,\
trials=bayes_trials,rstate=np.random.RandomState(50))

# sort the trials with lowest loss (highest AUC) first

#bayes_trials_results=sorted(bayes_trials,key=lambda x: x['loss'])
#bayes_trials_results = sorted(bayes_trials.results, key = lambda x: x['loss'])
#bayes_trials_results[:2]

results=pd.read_csv("gbm_trials.csv")

results.sort_values('loss', ascending = True, inplace = True)
results.reset_index(inplace = True, drop = True)

print("#------------------------------#")
print(results.head())
print("#------------------------------#")
print("\n")

import ast

# convert from a string to a dictionary

ast.literal_eval(results.loc[0,"params"])

# Evaluate best results

# extract the ideal number of estimators and hyperparameters

best_bayes_estimators=int(results.loc[0,"estimators"])
best_bayes_params=ast.literal_eval(results.loc[0,"params"]).copy()

# re-create the best model and train on the training data

#import lightgbm as lgb

best_bayes_model=lgb.LGBMClassifier(n_estimators=best_bayes_estimators,n_jobs=1,\
objective="binary",random_state=50,**best_bayes_params)

best_bayes_model.fit(features, labels)

# evaluate on the testing data

preds=best_bayes_model.predict_proba(test_features)[:,1]

print("#------------------------------#")
print('The best model from Bayes optimization scores {:.5f} AUC ROC on \
the test set.'.format(roc_auc_score(test_labels, preds)))
print('This was achieved after {} search iterations'.format(results.loc[0, 'iteration']))
print("#------------------------------#")
print("\n")

# Comparison to Random Search

# Optimal Hyperparameters

best_random_params["method"]="random search"
best_bayes_params["method"]="Bayesian optimization"

#best_params=pd.DataFrame(best_bayes_params,index=[0]).append(pd.DataFrame(\
#best_random_params,index=[0]),ignore_index=True,sort=True)

best_params=pd.DataFrame(best_bayes_params,index=[0]).append(pd.DataFrame(\
best_random_params,index=[0]),ignore_index=True)

print("#------------------------------#")
print(best_params)
print("#------------------------------#")
print("\n")

# visualizing hyperparameters

# create a new dataframe for storing parameters
# random search

random_params=pd.DataFrame(columns=list(random_results.loc[0,"params"].keys()),\
index=list(range(len(random_results))))

# add the results with each parameter a different column

for i,params in enumerate(random_results["params"]):
	random_params.loc[i,:]=list(params.values())
	
random_params["loss"]=random_results["loss"]
random_params["iteration"]=random_results["iteration"]

print("#------------------------------#")
print(random_params.head())
print("#------------------------------#")
print("\n")

# create a new dataframe for storing parameters
# bayesian search

bayes_params = pd.DataFrame(columns = list(ast.literal_eval(results.loc[0,\
'params']).keys()),index = list(range(len(results))))
for i, params in enumerate(results['params']):
	bayes_params.loc[i, :] = list(ast.literal_eval(params).values())
	
bayes_params['loss'] = results['loss']
bayes_params['iteration'] = results['iteration']

print("#------------------------------#")
print(bayes_params.head())
print("#------------------------------#")
print("\n")

# learning rate
# Density plots of the learning rate distributions

fig,axis=plt.subplots(figsize=(8,6))
#plt.style.use("fivethirtyeight")
sns.kdeplot(learning_rate_dist, label = 'Sampling Distribution', linewidth = 2)
sns.kdeplot(random_params['learning_rate'], label = 'Random Search', linewidth = 2)
sns.kdeplot(bayes_params['learning_rate'], label = 'Bayes Optimization', linewidth = 2)
plt.legend()
axis.set_xlabel("'Learning Rate")
axis.set_ylabel("Density")
axis.set_title("Learning Rate Distribution")
plt.savefig("figure-6.jpg",dpi=300)
plt.show()

# boosting type

fig,axis=plt.subplots(1,2,sharex=True,sharey=True)

# bar plots of boosting type
random_params['boosting_type'].value_counts().plot.bar(ax=axis[0],\
figsize=(14, 6),color='orange',title='Random Search Boosting Type')
bayes_params['boosting_type'].value_counts().plot.bar(ax=axis[1], \
figsize=(14, 6),color='green',title='Bayes Optimization Boosting Type')

plt.savefig("figure-7.jpg",dpi=300)
plt.show()

print("#------------------------------#")
print('Random Search boosting type percentages')
print(100 * random_params['boosting_type'].value_counts()/len(random_params))
print("\n")

print('Bayes Optimization boosting type percentages')
print(100 * bayes_params['boosting_type'].value_counts()/len(bayes_params))
print("\n")
print("#------------------------------#")
print("\n")

# Plots of All Numeric Hyperparameters

# Iterate through each hyperparameter

for i, hyper in enumerate(random_params.columns):
	if hyper not in ['class_weight', 'boosting_type', 'iteration', \
	'subsample', 'metric', 'verbose']:
		fig,axis=plt.subplots(figsize=(8,6))
		# Plot the random search distribution and the bayes search distribution
		if hyper != "loss":
			sns.kdeplot([sample(space[hyper]) for _ in range(1000)],\
			label = 'Sampling Distribution')
		sns.kdeplot(random_params[hyper],label='Random Search')
		sns.kdeplot(bayes_params[hyper],label='Bayes Optimization')
		plt.legend(loc = 1)
		axis.set_title("{} Distribution".format(hyper))
		axis.set_xlabel("{}".format(hyper))
		axis.set_ylabel("Density")
		plt.savefig("figure-8-{}.jpg".format(i),dpi=300)
		plt.show()
		
# evolution of hyperparameters searched

# map boosting type to integer (essentially label eccoding)

bayes_params["boosting_int"]=bayes_params["boosting_type"].replace({"gbdt":1,"goss":2,\
"dart":3})

# plot the boosting type over the serach

fig,axis=plt.subplots(figsize=(8,6))
axis.plot(bayes_params['iteration'], bayes_params['boosting_int'], 'ro')
axis.set_yticks([1, 2, 3], ['gdbt', 'goss', 'dart'])
axis.set_xlabel('Iteration')
axis.set_title('Boosting Type over Search')
plt.savefig("figure-9.jpg",dpi=300)
plt.show()

fig,axis=plt.subplots(1,4,figsize=(24, 6))

i=0

# plot of four hyperparameters

for i,hyper in enumerate(['colsample_bytree', 'learning_rate', \
'min_child_samples', 'num_leaves']):
	# scatter plot
	sns.regplot('iteration',hyper,data=bayes_params,ax=axis[i])
	axis[i].set(xlabel='Iteration',ylabel='{}'.format(hyper),\
	title = '{} over Search'.format(hyper))

plt.tight_layout()
plt.savefig("figure-10.jpg",dpi=300)
plt.show()

fig,axis = plt.subplots(1,3,figsize=(18,6))

i=0

# Scatter plot of next three hyperparameters

for i,hyper in enumerate(['reg_alpha','reg_lambda','subsample_for_bin']):
	sns.regplot('iteration',hyper,data=bayes_params,ax=axis[i])
	axis[i].set(xlabel='Iteration',ylabel='{}'.format(hyper),title='{} \
	over Search'.format(hyper))
	
plt.tight_layout()
plt.savefig("figure-11.jpg",dpi=300)
plt.show()

# validation losses

# DataFrame of just scores

scores = pd.DataFrame({'ROC AUC': 1 - random_params['loss'], 'iteration': \
random_params['iteration'], 'search': 'random'})
scores = scores.append(pd.DataFrame({'ROC AUC': 1 - bayes_params['loss'], \
'iteration': bayes_params['iteration'], 'search': 'Bayes'}))

scores['ROC AUC'] = scores['ROC AUC'].astype(np.float32)
scores['iteration'] = scores['iteration'].astype(np.int32)

print("#------------------------------#")
print(scores.head())
print("#------------------------------#")
print("\n")

plt.figure(figsize = (18, 6))

# Random search scores
plt.subplot(1, 2, 1)
plt.hist(1 - random_results['loss'].astype(np.float64), label = 'Random Search', edgecolor = 'k');
plt.xlabel("Validation ROC AUC"); plt.ylabel("Count"); plt.title("Random Search Validation Scores")
plt.xlim(0.75, 0.78)

# Bayes optimization scores
plt.subplot(1, 2, 2)
plt.hist(1 - bayes_params['loss'], label = 'Bayes Optimization', edgecolor = 'k');
plt.xlabel("Validation ROC AUC"); plt.ylabel("Count"); plt.title("Bayes Optimization Validation Scores");
plt.xlim(0.75, 0.78);

plt.savefig("figure-12.jpg",dpi=300)
plt.show()

# Plot of scores over the course of searching
sns.lmplot('iteration', 'ROC AUC', hue = 'search', data = scores, size = 8);
plt.xlabel('Iteration'); plt.ylabel('ROC AUC'); plt.title("ROC AUC versus Iteration");

plt.savefig("figure-13.jpg",dpi=300)
plt.show()

# save results

import json

with open("trials.json","w") as f:
	f.write(json.dumps(bayes_trials.results))
	
# Save dataframes of parameters

bayes_params.to_csv('bayes_params.csv',index = False)
random_params.to_csv('random_params.csv',index = False)

	
	
	

		
		
 



