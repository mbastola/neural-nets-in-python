import os
from utils import *
from ann import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

def trainAndTest(X,y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

    model.fit(X_train,y_train, batch_sz=10, epochs=10, debug=True)
    y_test = y_test.astype('float64')
    pred = (model.predict(X_test)).astype('float64')
    err = error_rate(pred,y_test)

    print("test error: " + str(err))
    print(confusion_matrix(y_test, pred))
    print("\n")
    print(classification_report(y_test,pred))

def testMnist(arch):    
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = scalePixel(X)
    y = strToIntArr(y)
    
    model = ANN("mnist10", arch )
    trainAndTest(X,y, model)
    
def testCifar10(arch):    
    X, y = fetch_openml('cifar_10', version=1, return_X_y=True)
    
    X = scalePixel(X)
    y = strToIntArr(y)

    print(X.shape)
    print(y.shape)

    model = ANN("cifar10", arch )
    trainAndTest(X,y, model)
    
def main():    
    #arch = [30, 10]
    arch2 = [128, 64, 32, 10]
    #testMnist(arch)
    testCifar10(arch2)    

if __name__ == "__main__":
    main()
