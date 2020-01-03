import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def error_rate(labels, predictions):
    return np.mean(labels != predictions)

def init_wb(M1,M2):
    W = np.random.randn(M1, M2) / np.sqrt(M1)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)


def init_filter(shape, poolsize):
    w = np.random.randn(*shape) * np.sqrt(2) / np.sqrt(np.prod(shape[:-1])+shape[-1]*np.prod(shape[:-2] / np.prod(poolsize)))
    return w.astype(np.float32)

def oneHotEncoder(X):
    N = len(X)
    K = len(set(X))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, X[i]] = 1
    return ind

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

def parseCNNInput(X,y):
    X = X.apply(lambda x: splitFloat(x))
    X = X.apply(lambda x: scalePixel(x))
    X = np.array(X)
    X = np.stack(X,axis=0)
    y = np.array(y)
    N, D = X.shape
    d = int(np.sqrt(D))
    X = X.reshape(N, 1, d, d) ## images are 48x48 = 2304 size vectors
    X = X.transpose((0, 2, 3, 1))
    return X,y
