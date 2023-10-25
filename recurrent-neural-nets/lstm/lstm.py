import numpy as np
from matplotlib import pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.fftpack import fft


#returns logistic map values for initial condition a and num steps n
def logistic_map(n, a):
    memo = [0]*(n)
    memo[0] = a
    for i in range(1, n):
        x_old = memo[i-1]
        memo[i] = 4*x_old*(1-x_old)
    return np.array(memo).reshape(-1,1)

#returns fiboncacci sequence of size n
def fibonacci(n):
    if n > 90:
        print("int overflow expected for n={}".format(n))
        return
    memo = [0] * n
    memo[1] = 1
    for i in range(2,n):
        memo[i] = memo[i-1]+memo[i-2]
    return np.array(memo).reshape(-1,1)

#returns fiboncacci sequence of size n mod prime number m
def fibonacci_mod_prime(n, m):
    memo = [0] * n
    memo[1] = 1
    for i in range(2,n):
        memo[i] = memo[i-1]+memo[i-2]
        if memo[i] > m:
            memo[i] %= m
    return np.array(memo).reshape(-1,1)

#returns sine values for initial condition a (phase) and num steps n and frequency f
def sine_map(n, a, f):
    t = np.arange(0, n)
    #T = 1.0 / n
    #x = np.linspace(0.0, n*T, n)
    #y = np.sin(2.0*np.pi* (x + a))
    return np.sin((t/f + a) * 2*np.pi).reshape(-1,1)

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(dataset[i + look_back,0])
    return np.array(dataX), np.array(dataY)


#run lstm training on 1d model
def lstm_train(dataset, scaler, look_back, epoch):
    #train_size = int(len(dataset) * 0.67)
    train_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    
    # create and fit the LSTM network
    model = Sequential()
    #model.add(LSTM(4, input_shape=(look_back, 1)))
    #model.add(Bidirectional(LSTM(4, input_shape=(look_back, 1))))
    model.add(Bidirectional(LSTM(8,  return_sequences=True, input_shape=(look_back, 1))))
    model.add(Bidirectional(LSTM(4)))
    #model.add(Dense(4))
    model.add(Dense(2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    hist = model.fit(trainX, trainY, epochs=epoch, batch_size=1, verbose=2)
    
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
        
    if scaler != None:
        # invert predictions

        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([trainY])
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY])

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY, testPredict))
    print('Test Score: %.2f RMSE' % (testScore))
        
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:] = np.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1] = testPredict
    
    plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
    # plot baseline and predictions
    if scaler != None:
        plt.plot(scaler.inverse_transform(dataset))
    else:
        plt.plot(dataset)
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.savefig("lstm_output.jpg")
    return model, trainPredict, testPredict,  hist.history["loss"]

"""
# normalize the dataset
dataset = b[:40]
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

out_plot = np.zeros(len(dataset)).reshape(-1,1)
plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
look_back = 2
out_plot[look_back:len(trainPredict)+look_back] = trainPredict
out_plot[len(trainPredict)+(look_back*2)+1:len(dataset)-1] = testPredict
plt.plot(out_plot)
plt.plot(c)
"""


#c = fibonacci_mod_prime(200, 113)/113.0
c = logistic_map(10000, 0.2)
dataset = c
#scaler = MinMaxScaler(feature_range=(0, 1))
#dataset = scaler.fit_transform(dataset)
model, trainPredict, testPredict, losses = lstm_train(dataset, None, 2, 15)
plt.clf()
plt.plot(losses)
plt.savefig("loses.jpg")

plt.clf()
look_back = 2
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back] = np.abs(trainPredict - c[look_back:len(trainPredict)+look_back])
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1] = np.abs(testPredict - c[len(trainPredict)+(look_back*2)+1:-1])

plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.savefig("diffs.jpg")
