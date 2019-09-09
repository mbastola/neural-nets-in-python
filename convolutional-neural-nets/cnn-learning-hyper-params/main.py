from utils import *
from classes import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix


def main():
    filename = "fer2013/fer2013.csv";
    df = pd.read_csv(filename)
    print(df.head())
    sns.countplot(x='emotion',data=df)
    
    # We start out with applying CNN on two class labels:0 (Angry) and  3 (Happy)

    binary = False  #read labels 3 and 4 only 
    balance = False
    if binary:
        df = df[(df['emotion']==3) | (df['emotion']==0)]
        df['emotion'] = df['emotion'].apply(lambda x: 0 if x == 0 else 1)

    X = df['pixels']
    y = df['emotion']
    
    if balance:
        maxLength = df.groupby('emotion').count()
        max_ = maxLength['Usage'].max()
        maxLength['times'] = maxLength['Usage'].apply(lambda x: np.ceil(max_*1.0/x))
        #print(maxLength)
        X = X.copy()
        y = y.copy()
        emot = 0;
        for item in maxLength['times']:
            for j in range(int(item)-1):
                X = pd.concat([X,df[df['emotion']==emot]['pixels']])
                y = pd.concat([y,df[df['emotion']==emot]['emotion']])
            emot+=1

    #sns.countplot(x=y)
    
    # The parseCNNInput takes (N,k) array and converts it into (N,c,sqrt(k),sqrt(k)) matrix for CNN. In this process, the pixel values are scaled to 0-1. 
    
    X,y = parseCNNInput(X,y)
    
    
    # Now we train a vanilla CNN with one 1 hidden layer and 2 pooling layers with default learning params
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    """
    model = CNN(architecture_params=
        {'pool_layer_sizes':[(20,5,5),(20,5,5)],
         'hidden_layer_sizes':[500,300]
        },
        hyper_params={'lr':0.1, 'mu':0.9, 'reg':0.01, 'decay':0.99, 'eps':0.0001} 
    )
    """

    model = CNN()
    model.fit(X_train,y_train,batch_sz=20,epochs=100,debug=True)
    pred = model.predict(X_test)
    err = error_rate(pred,y_test)
    print("test error: " + str(err))
    print(confusion_matrix(y_test, pred))
    print("\n")
    print(classification_report(y_test,pred))

    return
    
    # Lack of convergence shows that our parameter settings are not fine tuned. Lets learn the hyper paramters for this architecture. This includes learning_rate, momentum coefficient, regularizer parameter (l2), decay rate and epsilon. The way we will do this is to utilize scikit-learns Randomized Search to learn best paramters from 5D parameter space. Another way to do this would be is to use GridSearchCV which is an exhaustive search technique and would require long time to complete. 
    # 
    # We needed to make some modifications to the vanilla-cnn code to add get_params,set_params, score method. The estimator would fit the input data on the result (error_rate) of 1 epoch of learning which should be a good estimatore for the parameters we are seeking.
    # 
    # The randomized search took around 20 mins on Alienware R313. Its good to run it couple of times and take the average parameter values.  


    
    #Using Randomized Search for searching optimal learning parameters
    param_grid = {'lr':[0.00001,0.0005, 0.0001, 0.005, 0.001],'mu':[0.999,0.9945,0.99,0.945,0.9],'reg':[0.00001,0.00005,0.0001,0.0005,0.001],'decay':[0.999999,0.9999945,0.99999,0.999945,0.9999],'eps':[0.0001,0.0005,0.001,0.005,0.01]}

    estimator = RandomizedSearchCV(CNN(dropout_rates = [0.8, 0.5, 0.5]),param_grid, verbose=2)

    estimator.fit(X_train,y_train)

    print(estimator.best_params_)

    model = estimator.best_estimator_
    pred = model.predict(X_test)
    err = error_rate(pred,y_test)
    print("test error: " + str(err))
    print(confusion_matrix(y_test, pred))
    print("\n")
    print(classification_report(y_test,pred))

    # This CNN architecutre showed ~24% in training validation error while ~64% test error! It seems like the cnn is overfitting on the input data. One way to try better this result is to utilize dropouts. We modified the vanilla cnn to include dropouts as input and in fit function. Finally, I use p = 0.2 dropout for input layer while 0.5 drop-out for hidden layers. 


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
    model_with_dropout_reg = CNN(architecture_params=
        {'pool_layer_sizes':[(20,5,5),(20,5,5)],
         'hidden_layer_sizes':[500,300], 
         'dropout_rates':[0.8, 0.5, 0.5]
        },
        hyper_params= {'mu': 0.945, 'decay': 0.9999945, 'lr': 0.0005, 'reg': 0.0005, 'eps': 0.001}
    )



    model_with_dropout_reg.fit(X_train,y_train,batch_sz=30,epochs=6,debug=True)


    pred = model_with_dropout_reg.predict(X_test)
    err = error_rate(pred,y_test)
    print("test error: " + str(err))
    print(confusion_matrix(y_test, pred))
    print("\n")
    print(classification_report(y_test,pred))


    # We can see that adding dropout increased the training-validation error to ~31% but significantly reduced the testing error while ~36% test error, which is a good state to be.

if __name__ == "__main__":
    main()
