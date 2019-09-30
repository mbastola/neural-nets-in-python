"""
Manil Bastola
~~~~~~~~~~
original : https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py
"""

import random
from utils import *
from sklearn.utils import shuffle
import numpy as np

class CNET(object):

    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.sizes = layer_sizes
        #self.biases = [ np.random.randn(y, 1) for y in layer_sizes[1:]]
        #self.weights = [ np.random.randn(y, x)/np.sqrt(x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [(np.random.randn(y, 1) + 1j*np.random.randn(y, 1)) for y in layer_sizes[1:]]
        self.weights = [(np.random.randn(y, x) + 1j*np.random.randn(y, x))/np.sqrt(x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

    def forward(self, X):
        """Return the output of the network if ``X`` is input."""
        for b, w in zip(self.biases, self.weights):
            X = root(np.dot(w, X)+b)
        return np.square(X)

    def fit(self, X, Y,  batch_size, epochs, lr):
        """Train the neural network using mini-batch stochastic gradient descent"""
        n_valid = 1000
        validate = True
        X,Y = shuffle(X,Y)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]
        n = len(X)
        
        for j in range(epochs):
            X,Y = shuffle(X,Y)
            mini_batches = []
            for k in range(0, n, batch_size):
                X_batch = X[k: k+batch_size]
                Y_batch = Y[k: k+batch_size]
                self.update_mini_batch(X_batch, Y_batch, lr)
            if validate:
                Ypred = self.predict(Xvalid)
                error_valid = error_rate(oneHotDecoder(Yvalid), oneHotDecoder(Ypred))
                print("Epoch {0}: {1}".format(
                    j, error_valid* 100))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, X_batch, Y_batch, lr):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in zip(X_batch, Y_batch):
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(lr/len(X_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(lr/len(X_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = root(z)
            activations.append(activation)
        # backward pass
        activations[-1] = np.square(activations[-1])
        delta = self.cost_derivative(activations[-1], y) * root_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        first=1
        for l in range(2, self.num_layers):
            z = zs[-l]
            if first:
                z = 2*z
                first = 0
            sp = root_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def predict(self, X):
        test_results = [self.forward(x) for x in X]
        return np.array(test_results)
        
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
