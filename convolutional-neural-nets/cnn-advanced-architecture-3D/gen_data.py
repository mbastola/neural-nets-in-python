"""
Manil Bastola
Script Builds the LUNA image database as test and train datasets of numpy arrays
"""


import numpy as np
import pandas as pd

import os
import glob

import pickle
from sklearn.model_selection import train_test_split
from CTSData import *

base_dir = "/media/mbastola/Transcend/LUNA/data/"
candidates_file = 'candidates.csv'

def gen_data(idx, outDir, x_data, y_data,  width = 64):
    '''
    Generate training & testing data as numpy arrays of format N x d x data format where N is number of annotations, d is the resampled depth of the annotated scan & data is the unravled image_data. One can also get the train & test data as 2d images sing get_image_slice in the format N x 1 x data. 
    '''
    try:
        scan = CTSData(x_data[0], x_data[1:], y_data, raw_image_path)
        #return scan.get_image_slice()
        return scan.get_images(width)
    except Exception as e:
        print(e)
        return []

def split_test_train(filename):
    """
    Split training & testing data
    """
    df = pd.read_csv(filename)

    neg_annotations = df[df['class']==0].index
    pos_annotations  = df[df['class']==1].index
    
    ## Under Sample Negative Indexes for there are far too many negatives
    len_neg = neg_annotations.shape[0]
    len_pos = pos_annotations.shape[0]
    imbalance_ratio = len_pos*5.0/len_neg 
    
    neg_annotations = neg_annotations.sample(frac=imbalance_ratio).reset_index(drop=True)

    all_annotations = pd.concat([pos_annotations, neg_annotations])
    print("tot", all_annotations.info())
    X = all_annotations.drop(["class"])
    y = all_annotations[["class"]]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.20, random_state = 42)

    print(X_train.info())
    print(y_train.info())
    input()
    #save to folder
    X_train.to_pickle(base_dir + 'traindata')
    y_train.to_pickle(base_dir + 'trainlabels')
    X_test.to_pickle(base_dir + 'testdata')
    y_test.to_pickle(base_dir + 'testlabels')


def main():
    mode = 'train' #'test'
    
    inpmeta = base_dir + mode + 'data'
    inplabels = base_dir + mode + 'labels'
    outDir = base_dir + mode + '/image_'

    if not os.path.exists(inpmeta):
        do_test_train_split(candidates_file)

    X_data = pd.read_pickle(inpmeta)
    y_data = pd.read_pickle(inplabels)
    X_out = []
    y_out = []
    for i in range(0,10):
        X_out2 = []
        y_out2 = []
        #print("Folder: ",i)       
        global raw_image_path
        raw_image_path = 'subset' + str(i) + '/'
        for idx in X_data.index:
            label = y_data.loc[idx]
            img_1d = gen_data(idx, outDir, np.asarray(X_data.loc[idx]), y_data.loc[idx])
            if (len(img_1d) != 0):
                y_out.append(label)
                X_out.append(img_1d)
                y_out2.append(label)
                X_out2.append(img_1d)
        X_out2 = np.array(X_out2, dtype = np.float32)
        y_out2 = np.array(y_out2, dtype = np.float32)
        np.save(base_dir+mode+'_imdata'+str(i), X_out2)
        np.save(base_dir+mode+'_imlabel'+str(i), y_out2)

    X_out = np.array(X_out, dtype = np.float32)
    y_out = np.array(y_out, dtype = np.float32)
    print(X_out.shape)
    print(y_out.shape)
    np.save(base_dir+mode+'_imdata', X_out)
    np.save(base_dir+mode+'_imlabel', y_out)

if __name__ == "__main__":
    main()

        
