"""
Manil Bastola
Script compares CNN 2d to CNN3D for the LUNA dataset
"""


import os
from utils import *
#import cv2
from cnn3d import *
from cnn import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.utils import shuffle
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
    
def upsample_training_data(X_train, y_train):
    aug = []
    aug_y = []
    j = 0
    k = 0
    for i, each_data in enumerate(X_train):
        label = y_train[i]
        if label == 0:
            j+=1
            continue
        k+=1
        img = each_data.reshape(64,64)
        im = np.rot90(img)
        aug.append(im.ravel())
        aug_y.append(label)
        im = np.rot90(im)
        aug.append(im.ravel())
        aug_y.append(label)
        im = np.rot90(im)
        aug.append(im.ravel())
        aug_y.append(label)
        #im = cv2.GaussianBlur(im,(1,1),0)
        #aug.append(im.ravel())
        #aug_y.append(label)
    aug = np.array(aug)
    aug_y = np.array(aug_y)
    X_train_aug = np.concatenate([X_train, aug])
    y_train_aug = np.concatenate([y_train, aug_y])
    X_train_aug, y_train_aug = shuffle(X_train_aug, y_train_aug)
    return X_train_aug, y_train_aug

def get_luna2d():
    X_train_3d = np.load("./train_imdata.npy")
    X_test_3d = np.load("./test_imdata.npy")
    X_train = X_train_3d[:,5,:]
    X_test = X_test_3d[:,5,:]

    y_train = np.load("./train_imlabel.npy").astype(np.int32)
    y_test = np.load("./test_imlabel.npy").astype(np.int32)
    X_train_aug, y_train_aug = upsample_training_data(X_train, y_train)
    return X_train_aug,y_train_aug,X_test,y_test
    
def test_luna2d(arch):    
    X_train,y_train,X_test,y_test = get_luna2d()

    X_train = arrToImg(X_train)
    X_test = arrToImg(X_test)

    model = CNN("luna", architecture_params = arch )
    model.fit(X_train,y_train,batch_sz=300,epochs=30,method="adam", debug=True)
    pred = model.predict(X_test)
    err = error_rate(pred,y_test.astype('float64'))

    np.save("pred",pred)
    np.save("ytest",y_test)
    
    print("test error: " + str(err))
    print(confusion_matrix(y_test, pred))
    print("\n")
    print(classification_report(y_test,pred))


def upsample_training_data3d(X_train, y_train):
    aug = []
    aug_y = []
    j = 0
    k = 0
    for i, each_data in enumerate(X_train):
        label = y_train[i]
        if label == 0:
            j+=1
            continue
        k+=1
        img = each_data.reshape(-1, 64,64)
        im = np.rot90(img,1,(1,2))
        aug.append(im.reshape(-1, 4096))
        aug_y.append(label)
        im = np.rot90(img,1,(1,2))
        aug.append(im.reshape(-1, 4096))
        aug_y.append(label)
        im = np.rot90(img,1,(1,2))
        aug.append(im.reshape(-1, 4096))
        aug_y.append(label)
        
    aug = np.array(aug)
    aug_y = np.array(aug_y)
    X_train_aug = np.concatenate([X_train, aug])
    y_train_aug = np.concatenate([y_train, aug_y])
    X_train_aug, y_train_aug = shuffle(X_train_aug, y_train_aug)
    return X_train_aug, y_train_aug


def get_luna3d():
    X_train_3d = np.load("./train_imdata.npy")
    X_test_3d = np.load("./test_imdata.npy")

    y_train = np.load("./train_imlabel.npy").astype(np.int32)
    y_test = np.load("./test_imlabel.npy").astype(np.int32)
    X_train_3d, y_train = upsample_training_data3d(X_train_3d, y_train)
    return X_train_3d,y_train,X_test_3d,y_test
    
def test_luna3d(arch):    
    X_train,y_train,X_test,y_test = get_luna3d()
    X_train = arrToImg3d(X_train)
    print(X_train.shape)
    X_test = arrToImg3d(X_test)
    print(X_test.shape)

    model = CNN3D("luna3D", architecture_params = arch )
    model.fit(X_train,y_train,batch_sz=300,epochs=20,method="adam", debug=True)
    pred = model.predict(X_test,batch_sz=300)
    err = error_rate(pred,y_test.astype('float64'))

    np.save("pred",pred)
    np.save("ytest",y_test)
    
    print("test error: " + str(err))
    print(confusion_matrix(y_test, pred))
    print("\n")
    print(classification_report(y_test,pred))


def main():    
    arch = [
        {'type':'C','activation':'relu','num_output':64,'kernel_size': (3,3), 'stride': (1,1), 'drop_out': 8.3},
        {'type':'P','pool':'max','kernel_size': (2,2), 'stride': (2,2), 'drop_out': 16.7},
        {'type':'C','activation':'relu','num_output':128,'kernel_size': (3,3), 'stride': (1,1), 'drop_out': 0.0},
        {'type':'C','activation':'relu','num_output':64,'kernel_size': (5,5), 'stride': (1,1), 'drop_out':16.7},
        {'type':'P','pool':'max','kernel_size': (5,5), 'stride': (3,3), 'drop_out': 0.0},
        {'type':'C','activation':'relu','num_output':128,'kernel_size': (3,3), 'stride': (1,1), 'drop_out': 0.0},
        {'type':'FC','activation':'relu','num_output':20,'drop_out': 8.3},
        {'type':'FC','activation':'relu','num_output':10,'drop_out': 0.0}, 
        {'type':'T','activation':None}
    ]

    archsm = [{'type':'C','activation':'relu','num_output':16,'kernel_size': (3,3), 'stride': (1,1), 'drop_out': 8.3},
        {'type':'P','pool':'max','kernel_size': (2,2), 'stride': (2,2), 'drop_out': 16.7},
        {'type':'C','activation':'relu','num_output':32,'kernel_size': (3,3), 'stride': (1,1), 'drop_out': 0.0},
        {'type':'C','activation':'relu','num_output':16,'kernel_size': (5,5), 'stride': (1,1), 'drop_out':16.7},
        {'type':'P','pool':'max','kernel_size': (5,5), 'stride': (3,3), 'drop_out': 0.0},
        {'type':'C','activation':'relu','num_output':32,'kernel_size': (3,3), 'stride': (1,1), 'drop_out': 0.0},
        {'type':'FC','activation':'relu','num_output':20,'drop_out': 8.3},
        {'type':'FC','activation':'relu','num_output':10,'drop_out': 0.0}, 
        {'type':'T','activation':None}
    ]

    
    arch3d = [
        {'type':'C','activation':'relu','num_output':64,'kernel_size': (3,3,3), 'stride': (1,1,1), 'drop_out': 8.3},
        {'type':'P','pool':'max','kernel_size': (2,2,2), 'stride': (2,2,2), 'drop_out': 16.7},
        {'type':'C','activation':'relu','num_output':128,'kernel_size': (3,3,3), 'stride': (1,1,1), 'drop_out': 0.0},
        {'type':'C','activation':'relu','num_output':64,'kernel_size': (5,5,1), 'stride': (1,1,1), 'drop_out':16.7},
        {'type':'P','pool':'max','kernel_size': (5,5,2), 'stride': (3,3,3), 'drop_out': 0.0},
        {'type':'C','activation':'relu','num_output':128,'kernel_size': (3,3,3), 'stride': (1,1,1), 'drop_out': 0.0},
        {'type':'FC','activation':'relu','num_output':20,'drop_out': 8.3},
        {'type':'FC','activation':'relu','num_output':10,'drop_out': 0.0}, 
        {'type':'T','activation':None}
    ]

    arch3dsm = [
        {'type':'C','activation':'relu','num_output':16,'kernel_size': (3,3,3), 'stride': (1,1,1), 'drop_out': 8.3},
        {'type':'P','pool':'max','kernel_size': (2,2,2), 'stride': (2,2,2), 'drop_out': 16.7},
        {'type':'C','activation':'relu','num_output':32,'kernel_size': (3,3,3), 'stride': (1,1,1), 'drop_out': 0.0},
        {'type':'C','activation':'relu','num_output':16,'kernel_size': (5,5,1), 'stride': (1,1,1), 'drop_out':16.7},
        {'type':'P','pool':'max','kernel_size': (5,5,2), 'stride': (3,3,3), 'drop_out': 0.0},
        {'type':'C','activation':'relu','num_output':32,'kernel_size': (3,3,3), 'stride': (1,1,1), 'drop_out': 0.0},
        {'type':'FC','activation':'relu','num_output':20,'drop_out': 8.3},
        {'type':'FC','activation':'relu','num_output':10,'drop_out': 0.0}, 
        {'type':'T','activation':None}
    ]

    #test_luna2d(archsm)
    test_luna3d(arch3dsm)
    
if __name__ == "__main__":
    main()
