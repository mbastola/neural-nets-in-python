import os
from utils import *
from cnn import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

def trainAndTest(X,y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

    #model.fit(X_train,y_train, batch_sz=10, epochs=10, debug=True)
    model.fit(X_train,y_train,batch_sz=1500,epochs=30,method="adam", debug=True)
        
    y_test = y_test.astype('float64')
    pred = (model.predict(X_test)).astype('float64')
    err = error_rate(pred,y_test)

    print("test error: " + str(err))
    print(confusion_matrix(y_test, pred))
    print("\n")
    print(classification_report(y_test,pred))

def testMnist(arch=None):    
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = scalePixel(X)
    X = arrToImg(X)    
    y = strToIntArr(y)
    
    model = CNN("mnist",fc_layer_sizes = arch)
    trainAndTest(X,y, model)
    
def testCifar10(arch):    
    X, y = fetch_openml('cifar_10', version=1, return_X_y=True)
    
    X = scalePixel(X)
    X = arrToImg(X,3)    
    y = strToIntArr(y)

    model = CNN("cifar10")
    trainAndTest(X,y, model)
    
def main():    
    arch = [30]
    arch2 = [128, 64, 32]
    #testMnist(arch)
    testCifar10(arch2)    

if __name__ == "__main__":
    main()
