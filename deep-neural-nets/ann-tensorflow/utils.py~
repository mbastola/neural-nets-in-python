#import cPickle
import _pickle as cPickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))                

def error_rate(labels, predictions):
    return np.mean(labels != predictions)

def oneHotEncoder(X):
    N = len(X)
    K = len(set(X))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, X[i]] = 1
    return ind

def oneHotDecoder(X):
    N = len(X)
    ind = np.zeros(N)
    for i in range(N):
        ind[i] = np.argmax(X[i])
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
    
def offsetX(x):
    i = 0
    y = []
    for item in x:
        y.append(np.reshape(item,(item.shape[0],1)))
        i=i+1
    return np.array(y)
    
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
