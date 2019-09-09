#import cPickle
import _pickle as cPickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

def error_rate(labels, predictions):
    return np.mean(labels != predictions)

def oneHotEncoder(X):
    N = len(X)
    K = len(set(X))
    print(N,K)
    input()
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, X[i]] = 1
    return ind

def uniqueStr():
    nowinsec = int((datetime.now()-datetime(1970,1,1)).total_seconds())
    return str(nowinsec)
  
def splitFloat(x):
    splitted = x.split(" ")
    i = 0
    for item in splitted:
        splitted[i]=float(item)
        i=i+1
    return splitted
    
def scalePixel(x):
    i = 0
    for item in x:
        x[i] = item/255.0
        i=i+1
    return np.array(x)
    
def dftoArr(X,y):
    X = X.apply(lambda x: splitFloat(x))
    X = X.apply(lambda x: scalePixel(x))
    X = np.array(X)
    X = np.stack(X,axis=0)
    y = np.array(y)
    return X,y

def strToIntArr(X):
    N = len(X)
    for i in range(N):
        X[i] = int(X[i])
    return np.array(X) #.astype('float64')


def arrToImg(X, ch=1):
    N, D = X.shape
    d = int(np.sqrt(D/ch))
    X = X.reshape(N, ch, d, d)
    X = X.transpose((0, 2, 3, 1))
    return X

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='bytes')
    return dict
"""
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict
"""
def balanceClasses(X,y):
    maxLength = df.groupby('emotion').count()
    max_ = maxLength['Usage'].max()
    maxLength['times'] = maxLength['Usage'].apply(lambda x: np.ceil(max_*1.0/x))
    #print(maxLength)
    X_ = X.copy()
    y_ = y.copy()
    emot = 0;
    for item in maxLength['times']:
        for j in range(int(item)-1):
            X_ = pd.concat([X_,df[df['emotion']==emot]['pixels']])
            y_ = pd.concat([y_,df[df['emotion']==emot]['emotion']])
        emot+=1
    return X_,y_


###for Q learning

def max_dict(d):
    # returns the argmax (key) and max (value) from a dictionary
    # put this into a function since we are using it so often
    max_key = None
    max_val = float('-inf')
    for k, v in d.iteritems():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val

def random_action(a, eps=0.1):
  # we'll use epsilon-soft to ensure all states are visited
  # what happens if you don't do this? i.e. eps=0
  p = np.random.random()
  if p < (1 - eps):
    return a
  else:
    return np.random.choice(ALL_POSSIBLE_ACTIONS)

def print_values(V, g):
  for i in xrange(g.width):
    print("---------------------------")
    for j in xrange(g.height):
      v = V.get((i,j), 0)
      if v >= 0:
        print(" %.2f|" % v,)
      else:
        print ("%.2f|" % v,) # -ve sign takes up an extra space


def print_policy(P, g):
  for i in xrange(g.width):
    print ("---------------------------")
    for j in xrange(g.height):
      a = P.get((i,j), ' ')
      print ("  %s  |" % a,)
    


