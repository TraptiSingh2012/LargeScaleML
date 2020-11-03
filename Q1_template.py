import matplotlib.pyplot as plt
import random
import time
import numpy as np
from sklearn import metrics
from math import sqrt
from sklearn.model_selection import train_test_split\

def diffa(y, ypred,x):
    return (y-ypred)*(-x)

def diffb(y, ypred):
    return (y-ypred)*(-1)

def shuffle_data(x,y):
    # shuffle x，y，while keeping x_i corresponding to y_i
    seed = random.random()
    random.seed(seed)
    random.shuffle(x)
    random.seed(seed)
    random.shuffle(y)

def get_batch_data(x, y, batch):
    shuffle_data(x, y)
    x_batch = x[0:batch]
    y_batch = y[0:batch]
    return [x_batch, y_batch]

data = np.loadtxt('LinearRegdata.txt')
x = data[:, 1]
y = data[:, 2]
for i in range(0,5):
  print("x[",i,"] = ",x[i],",","y[",i,"] = ",y[i])

# Normalize the data
x_max = max(x)
x_min = min(x)
y_max = max(y)
y_min = min(y)
for i in range(0, len(x)):
    x[i] = (x[i] - x_min)/(x_max - x_min)
    y[i] = (y[i] - y_min)/(y_max - y_min)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=109)


def batch_gd(X_train, y_train):

	a = 10.0
	b = -20.0

	all_bgdloss = []
	all_ep = []

	rate = 0.008
	start = time.time()
	for ep in range(1,100):
	    loss = 0
	    losst = 0
	    all_da = 0
	    all_db = 0
	    for i in range(0, len(X_train)):
	        y_pred = a*X_train[i] + b
	        loss = loss + (y_train[i] - y_pred)*(y_train[i] - y_pred)/2
	        all_da = all_da + diffa(y_train[i], y_pred, X_train[i]) #gradients accumulated
	        all_db = all_db + diffb(y_train[i], y_pred) #gradients accumulated

	    loss = loss/len(X_train)
	    all_bgdloss.append(loss)
	    all_ep.append(ep)

	    #parameters updated
	    a = a - rate * all_da
	    b = b - rate * all_db

	    for i in range(0, len(X_test)):
	        y_pred = a*X_test[i] + b
	        losst = losst + (y_test[i] - y_pred)*(y_test[i] - y_pred)/2

	    #Saving best parameters for final reporting on test set   
	    if ep==1:
	        prevloss = losst
	    else:
	        if losst<prevloss:
	          prevloss=losst
	          param1 = a
	          param2 = b


def minibatch_gd():
	pass

def stochastic_gd():
	pass

def momentum_gd():
	pass

def adam_gd():
	pass

def main():

	batch_gd(X_train, y_train)

if __name__=='__main__':
	main()