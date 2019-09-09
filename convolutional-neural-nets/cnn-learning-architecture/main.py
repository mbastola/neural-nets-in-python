from utils import *
from cnn import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

def main():
    
    filename = "../../data/fer2013/fer2013.csv";
    df = pd.read_csv(filename)
    #sns.countplot(x='emotion',data=df)

    # We start out with applying CNN on two class labels:0 (Angry) and  3 (Happy)

    #input_data.read_data_sets("MNIST_data/", one_hot=True)
    binary = True  #read labels 3 and 4 only 
    balance = False
    if binary:
        df = df[(df['emotion']==3) | (df['emotion']==0)]
        df['emotion'] = df['emotion'].apply(lambda x: 0 if x == 0 else 1)
    X = df['pixels']
    y = df['emotion']
    if balance:
        X, y = balanceClasses(X,y)
    #sns.countplot(x=y)

    # The parseCNNInput takes (N,k) array and converts it into (N,c,sqrt(k),sqrt(k)) matrix for CNN. In this process, the pixel values are scaled to 0-1. 

    X,y = dftoArr(X,y)
    X = arrToImg(X)
    
    """
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = scalePixel(X)
    X = arrToImg(X)
    Y = strToIntArr(y)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

    #hyper_params={'lr':0.1, 'mu':0.9, 'reg':0.01, 'decay':0.99, 'eps':0.0001}

    arch1 = [
        {'type':'C','activation':'relu','num_output':64,'kernel_size': (3,3), 'stride': (1,1), 'drop_out': 0.0},
        {'type':'C','activation':'relu','num_output':256,'kernel_size': (3,3), 'stride': (1,1), 'drop_out': 8.3},
        {'type':'P','pool':'max','kernel_size': (2,2), 'stride': (2,2), 'drop_out': 0.0},
        {'type':'C','activation':'relu','num_output':512,'kernel_size': (3,3), 'stride': (1,1), 'drop_out': 16.7},
        {'type':'C','activation':'relu','num_output':256,'kernel_size': (1,1), 'stride': (1,1), 'drop_out': 16.7},
        {'type':'P','pool':'max','kernel_size': (5,5), 'stride': (3,3), 'drop_out': 25.0},
        {'type':'C','activation':'relu','num_output':256,'kernel_size': (3,3), 'stride': (1,1), 'drop_out': 0.0},
        {'type':'C','activation':'relu','num_output':512,'kernel_size': (1,1), 'stride': (1,1), 'drop_out': 33.3},
        {'type':'FC','activation':'relu','num_output':512,'drop_out': 41.7},
        {'type':'FC','activation':'relu','num_output':64,'drop_out': 0.0},
        {'type':'T','activation':'softmax'}
    ]

    arch2 = [
        {'type':'C','activation':'relu','num_output':64,'kernel_size': (3,3), 'stride': (1,1), 'drop_out': 0.0},
        {'type':'C','activation':'relu','num_output':128,'kernel_size': (3,3), 'stride': (1,1), 'drop_out': 8.3},
        {'type':'P','pool':'max','kernel_size': (2,2), 'stride': (2,2), 'drop_out': 16.7},
        {'type':'C','activation':'relu','num_output':256,'kernel_size': (3,3), 'stride': (1,1), 'drop_out': 0.0},
        {'type':'C','activation':'relu','num_output':128,'kernel_size': (5,5), 'stride': (1,1), 'drop_out':16.7},
        {'type':'P','pool':'max','kernel_size': (5,5), 'stride': (3,3), 'drop_out': 0.0},
        {'type':'C','activation':'relu','num_output':256,'kernel_size': (3,3), 'stride': (1,1), 'drop_out': 0.0},
        {'type':'FC','activation':'relu','num_output':256,'drop_out': 8.3},
        {'type':'FC','activation':'relu','num_output':128,'drop_out': 0.0}, 
        {'type':'T','activation':None}
    ]

    #{'type':'FC','activation':'tanh','num_output':128,'drop_out': 0.0},

    model = CNN("fer2013", architecture_params = arch2 )
    model.fit(X_train,y_train,batch_sz=100,epochs=2,method="adam", debug=True)
    pred = model.predict(X_test)
    err = error_rate(pred,y_test)
    print("test error: " + str(err))
    print(confusion_matrix(y_test, pred))
    print("\n")
    print(classification_report(y_test,pred))


if __name__ == "__main__":
    main()
