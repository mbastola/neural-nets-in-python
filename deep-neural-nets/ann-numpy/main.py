import os
from utils import *
from ann import *
from cnet import *
from net import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

def trainAndTest(X,y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

    model.fit(X_train,y_train,batch_size=100,epochs=10, lr=3.0)
    pred = model.predict(X_test)
    err = error_rate(pred,y_test.astype('float64'))

    np.save("pred",pred)
    np.save("ytest",y_test)
    
    print("test error: " + str(err))
    print(confusion_matrix(y_test, pred))
    print("\n")
    print(classification_report(y_test,pred))


def testfer2013(arch, binary=True):
    datadir = os.environ["ML_DATASETS_DIR"]
    filename = datadir+"/fer2013/fer2013.csv";
    df = pd.read_csv(filename)

    #read labels 3 and 4 only 
    if binary:
        df = df[(df['emotion']==3) | (df['emotion']==0)]
        df['emotion'] = df['emotion'].apply(lambda x: 0 if x == 0 else 1)
    X = df['pixels']
    y = df['emotion']
    #sns.countplot(x=y)

    # The parseCNNInput takes (N,k) array and converts it into (N,c,sqrt(k),sqrt(k)) matrix for CNN. In this process, the pixel values are scaled to 0-1. 

    X,y = dftoArr(X,y)
    X = arrToImg(X)
    
    model = CNN("fer2013", architecture_params = arch )
    trainAndTest(X,y, model)


def testMnist(arch):    
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = scalePixel(X)
    X = offsetX(X)
    #X = [np.reshape(x, (784, 1)) for x in X]
    #X = arrToImg(X)
    y = strToIntArr(y)
    y = oneHotEncoder(y)
    y = offsetX(y)
    model = CNET(layer_sizes = arch )
    #model = ANN(layer_sizes = arch )
    trainAndTest(X,y, model)
    #    model = Network(sizes = arch )
    #trainAndTest2(X,y, model)

def trainAndTest2(X,y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    train_set = (X_train,y_train)
    test_set = (X_test,y_test)
    
    train_set = [( X_train[i],y_train[i]) for i in range(X_train.shape[0])] 
    test_set = [( X_test[i], y_test[i]) for i in range(X_test.shape[0])]

    #test_set = list((X_test,y_test))
    #test_set = np.concatenate((X_test,y_test), axis=1)
    #print(X_train.shape)
    #print(y_train.shape)
    #print(X_test.shape)
    #print(y_test.shape)
    #print(test_set.shape)
    #input()

    model.SGD(list(train_set),epochs=30, mini_batch_size=10, eta=3.0, test_data = test_set )
    err = model.evaluate(list(test_set))
    print("test error: " + str(err))
    


def testCifar10(arch):    
    X, y = fetch_openml('cifar_10', version=1, return_X_y=True)
    
    X = scalePixel(X)
    X = arrToImg(X,3)
    y = strToIntArr(y)

    print(X.shape)
    print(y)
    input()
    model = CNN("cifar10", architecture_params = arch )
    trainAndTest(X,y, model)

def testCifar100(arch):

    datadir = os.environ["ML_DATASETS_DIR"]
    filename1 = datadir+"/cifar100/train";
    filename2 = datadir+"/cifar100/test";

    train_dict = unpickle(filename1)
    test_dict = unpickle(filename1)

    X = train_dict.get("data".encode())
    y = train_dict.get("fine_labels".encode())

    
    X_test = test_dict.get("data".encode())
    y_test = test_dict.get("fine_labels".encode())

    X = np.concatenate((X,X_test))
    y = np.concatenate((y,y_test))
    
    X = scalePixel(X)
    X = arrToImg(X,3)
    y = strToIntArr(y)

    print(X.shape)
    print(y.shape)
    
    input()
    model = CNN("cifar100", architecture_params = arch )
    trainAndTest(X,y, model)

def testSHVN(arch):

    X, y = fetch_openml('shvn', return_X_y=True)
    
    X = scalePixel(X)
    X = arrToImg(X,3)
    y = strToIntArr(y)

    
    print(X.shape)
    print(y)
    input()
    model = CNN("shvn", architecture_params = arch )
    trainAndTest(X,y, model)


def main():    
    arch = [784, 30, 10]
    #testfer2013(arch2)
    #testfer2013(arch2,False)
    testMnist(arch)
    #testCifar10(arch2)
    #testCifar100(arch2)
    #testSHVN(arch2)
    

if __name__ == "__main__":
    main()
