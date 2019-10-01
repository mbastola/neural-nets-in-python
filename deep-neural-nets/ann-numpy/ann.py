"""
Manil Bastola
~~~~~~~~~~
SKlearn type class with fit/predict functions for ANN. Based on the examples from the book "Neural Networks and Deep Learning": http://neuralnetworksanddeeplearning.com/
"""

import random
from utils import *
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

class ANN(object):

    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.sizes = layer_sizes
        self.biases = None
        self.weights = None
        
    def build(self):
        self.biases = [np.random.randn(x, 1) for x in self.sizes[1:]]
        self.weights = [(np.random.randn(y, x))/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
    
    def forward(self, X):
        for b, w in zip(self.biases, self.weights):
            X = sigmoid(np.dot(w, X)+b)
        return X

    def fit(self, X, Y, batch_size, epochs, lr, debug):
        """Using mini-batch SGD"""

        n_valid = 1000
        X,Y = shuffle(X,Y)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]
        n,f,_ = X.shape
        #get num features from input
        self.sizes[0] = f
        #initiate weights and biases
        self.build()
        
        losses = []
        train_errors = []
        for j in range(epochs):
            X,Y = shuffle(X,Y)
            mini_batches = []
            for k in range(0, n, batch_size):
                X_batch = X[k: k+batch_size]
                Y_batch = Y[k: k+batch_size]
                self.mini_batch_sgd(X_batch, Y_batch, lr)

            #validate
            Ypred = self.predict(Xvalid)
            error_valid = error_rate(oneHotDecoder(Yvalid), oneHotDecoder(Ypred))
            train_errors.append(error_valid)
            if debug:
                print("Epoch {0}: {1}".format(j, error_valid*100))
        if debug:
            #fig, axes = plt.subplots(nrows=1,ncols=2) 
            #axes[0].plot(losses)
            #axes[1].plot(train_errors)
            #plt.tight_layout()
            #plt.show()
            plt.plot(train_errors)
            plt.savefig("imgs/ann1-errors.png")
        print ('\n training error_rate: ', train_errors[-1])
            
    def mini_batch_sgd(self, X_batch, Y_batch, lr):
        """Update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in zip(X_batch, Y_batch):
            #print(x.shape)
            #print(y)
            #input()
            #x = np.reshape(x, (784, 1))
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(lr/len(X_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(lr/len(X_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a gradient for the cost function C with respect to self.weights and self.biases"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        activations[-1] = np.real(activations[-1])
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def predict(self, X):
        #return np.argmax(self.forward(X.transpose()))
        test_results = [self.forward(x) for x in X]
        return np.array(test_results)
        

    def cost_derivative(self, output_activations, y):
        """Partial derivative of cost C wrt a as difference"""
        return (output_activations-y)
