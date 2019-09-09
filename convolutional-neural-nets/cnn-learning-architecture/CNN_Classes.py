from utils import *
from layers import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle


class TerminalLayer(object):
    #final activation
    #_i = layer depth/layer-id
    #a = activation function
    def __init__(_i, param):
        self.layer_depth = _i
        self.activation_type = param['a']

    def forward(self,X):
        if self.activation_type == "global avg. pool":
            return tf.nn.relu(X)
        elif self.activation_type == "softmax":
            return tf.nn.tanh(X)

#Z = tf.nn.dropout(Z, self.dropout_rates[0])        
class DOLayer(object):
    #Drop Out
    #_i = layer depth/layer-id
    #r = Drop Out rate
    def __init__(_i, param):
        self.layer_depth = _i
        self.dropout_rate = param['r']

    def forward(self,X):
        return tf.nn.dropout(X, self.dropout_rate)  

class FCLayer(object):
    #Fully Connected layer
    #_i = layer depth/layer-id
    #mi = number of input features
    #mo = number of output features
    #a = activation function
    def __init__(self, param):
        _i = param['i']
        mi = param['mi']
        mo = param['mo']
        a =  param['a']
        self.layer_depth = _i
        self.W = tf.Variable(tf.truncated_normal([mi, mo])
        self.b = tf.Variable(tf.truncated_normal([mo]))
        
    def forward(self, X):
        out = tf.matmul(X, self.W) + self.b
        if self.a == None:
            return out
        elif self.a == "relu":
            return tf.nn.relu(out)
        elif self.a == "tanh":
            return tf.nn.tanh(out)
        elif self.a == "sigmoid":
            return tf.nn.sigmoid(out)

        
class ConvLayer(object):
    #Convolution Layer
    #_i = layer depth
    #mi = num_input features
    #mo = num_output_features
    #ksize = convolution kernel_size
    #l = strides
    #a = activation function    
    def __init__(self, param):
        _i = param['i']
        mi = param['mi']
        mo = param['mo']
        a =  param['a']
        l = param['l']
        ksize = param['ksize']

        self.layer_depth = _i
        self.l = l
        self.a = a
        fw, fh = ksize
        self.W = tf.Variable(tf.truncated_normal([fw, fh, mi, mo], stddev=0.03))
        self.b = tf.Variable(tf.truncated_normal([mo]))
        
    def forward(self, X):
        sw, sh = self.l
        conv_out = tf.nn.conv2d(X, self.W, strides=[1, sw, sh, 1], padding='SAME')
        conv_out = tf.nn.bias_add(conv_out, self.b)
        if self.a == None:
            return conv_out
        elif self.a == "relu":
            return tf.nn.relu(conv_out)
        elif self.a == "tanh":
            return tf.nn.tanh(conv_out)
        elif self.a == "sigmoid":
            return tf.nn.sigmoid(conv_out)

    #f = receptive field size
    #d = num_receptive_fields
    #n = representation size
    def getRepresentations(self):
        return (0,0,0)                     
        
class PoolLayer(object):
    #Pooling Layer
    #ksize = pooling kernel_size
    #l = strides
    def __init__(self, _i, ksize = (2,2), l = (2,2)):
        _i = param['i']
        l = param['l']
        ksize = param['ksize']
        
        self.layer_depth = _i
        self.l = l
        self.ksize = ksize
    
    def forward(self, X):
        sw, sh = self.l
        fw, fh = self.ksize
        pool_out = tf.nn.max_pool(
            X,
            ksize=[1, fw, fh, 1],
            strides=[1, sw, sh, 1],
            padding='SAME'
        )
        return pool_out
    
class CNN(object):
    def __init__(self, architecture_params, learning_params={'lr':1e-3, 'mu':0.99, 'reg':1e-3, 'decay':0.99999, 'eps':1e-10}):
        #architecture_params: [C,C,P,C,P,FC,T]
        #learning_params: [learning_rate, momentum, regularizer, decay_rate, epsilon]
        self.arch_params = architecture_params
        self.learning_params = learning_params
        self.layers = []
        self.K = 2

    def initLayer(self, layer_type, layer_param):
        if layer_type == 'C':
            return ConvLayer(layer_param)
        elif layer_type == 'P':
            return PoolLayer(layer_param)
        elif layer_type == 'FC':
            return FCLayer(layer_param)
        elif layer_type == 'T':
            return TerminalLayer(layer_param)
                             
    def build(self, initial_shape):
        N, width, height, ch = initial_shape
        mi = ch
        outw = width
        outh = height

        for layer_type in self.architecture_params:
            layer_param = self.architecture_params[layer_type]
            #layer_param['mi'] = mi
            layer = self.initLayer(layer_type, layer_param)
            self.layers.append(layer)
        
            

                             
    def fit(self, X, Y, batch_sz=500, epochs=1, method='adam',debug=False):
        lr = np.float32(self.learning_params['lr'])
        mu = np.float32(self.learning_params['mu'])
        reg = np.float32(self.learning_params.['reg'])
        decay = np.float32(self.learning_params.['decay'])
        eps = np.float32(self.learning_params.['eps'])
        self.K = len(set(Y))
        
        # make a validation set
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Y = oneHotEncoder(Y).astype(np.float32)
        
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]
        Yvalid_flat = np.argmax(Yvalid, axis=1) # for calculating error rate
        # initialize the architecture
        self.build(X.shape)                             

        # initialize mlp layers
        M1 = self.pool_layer_sizes[-1][0]*outw*outh # size must be same as output of last convpool layer
        count = 0
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1

        # logistic regression layer
        W, b = init_wb(M1, self.K)
        self.W = tf.Variable(W, 'W_logreg')
        self.b = tf.Variable(b, 'b_logreg')

        # collect params for later use
        self.params = [self.W, self.b]
        for h in self.pool_layers:
            self.params += h.params
        for h in self.hidden_layers:
            self.params += h.params

        # set up tensorflow functions and variables
        tfX = tf.placeholder(tf.float32, shape=(None, width, height, c), name='X')
        tfY = tf.placeholder(tf.float32, shape=(None, self.K), name='Y')
        act = self.forward(tfX,self.drop_out_regularizer)

        rcost = reg*sum([tf.nn.l2_loss(p) for p in self.params])
    
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=act,
                labels=tfY
            )
        ) + rcost

        
        prediction = self.predict(tfX)
        
        if (method=="rms"):
            train_op = tf.train.RMSPropOptimizer(lr, decay=decay, momentum=mu).minimize(cost)
        elif (method=="momentum"):
            train_op = tf.train.MomentumOptimizer(lr, momentum=mu).minimize(cost)
        elif (method=="adam"):
            train_op = tf.train.AdamOptimizer(lr).minimize(cost)
        else:
            print("gd cost minimizer unkown")
        n_batches = N // batch_sz
        costs = []
        train_errors = []
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            for i in range(epochs):
                X, Y = shuffle(X, Y)
                for j in range(n_batches):
                    Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                    Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

                    session.run(train_op, feed_dict={tfX: Xbatch, tfY: Ybatch})

                    if j == (n_batches-1):
                        c = session.run(cost, feed_dict={tfX: Xvalid, tfY: Yvalid})
                        costs.append(c)

                        p = session.run(prediction, feed_dict={tfX: Xvalid, tfY: Yvalid})
                        e = error_rate(Yvalid_flat, p)
                        train_errors.append(e)
                        if debug:
                            print("i:", i, "j:", j, "nb:", "cost:", c, "error rate:", e)

        if debug:
            fig, axes = plt.subplots(nrows=1,ncols=2) 
            axes[0].plot(costs)
            axes[1].plot(train_errors)
            plt.tight_layout()
        print ('\nerror_rate: ', train_errors[-1], ', params: ', {'lr':self.lr,'mu':self.mu,'reg':self.reg,'decay': self.decay,'eps':self.eps})

    def forward(self, X, drop_out=False):
        Z = X
        if drop_out:
            Z = tf.nn.dropout(Z, self.dropout_rates[0])
        for c in self.pool_layers:
            Z = c.forward(Z)
        Z_shape = Z.get_shape().as_list()
        Z = tf.reshape(Z, [-1, np.prod(Z_shape[1:])])
        if drop_out:
            for h, p in zip(self.hidden_layers, self.dropout_rates[1:]):
                Z = h.forward(Z)
                Z = tf.nn.dropout(Z, p)
        else:
            for h in self.hidden_layers:
                Z = h.forward(Z)
        return tf.matmul(Z, self.W) + self.b

    def predict(self, X):
        if (str(type(X)) == "<class 'numpy.ndarray'>"):
            init = tf.global_variables_initializer()
            with tf.Session() as session:
                session.run(init)
                N, width, height,c = X.shape
                tfX = tf.placeholder(tf.float32, shape=(None, width, height, c), name='X')
                #Y = tf.placeholder(tf.float32, shape=(None, self.K), name='Y')
                pY = self.forward(tfX)
                prediction = tf.argmax(pY, 1)
                p = session.run(prediction, feed_dict={tfX: X})
                print("xx ",np.unique(p))
                input()
                return p
        else:
            pY = self.forward(X,self.drop_out_regularizer)
            return tf.argmax(pY, 1)
    
    def score(self, X,y):
        pred = self.predict(X)
        return np.mean(y == pred)
    
    def get_params(self,deep=True):
        #hyper_params: [learning_rate, momentum, regularizer, decay_rate, epsilon]
        return {'lr':self.lr,'mu':self.mu,'reg':self.reg,'decay': self.decay,'eps':self.eps}
        #return {'architecture_params':{'pool_layer_sizes':self.pool_layer_sizes,'hidden_layer_sizes':self.hidden_layer_sizes,'dropout_rates':self.dropout_rates},'hyper_params':{'lr':self.lr,'mu':self.mu,'reg':self.reg,'decay': self.decay,'eps':self.eps}}
    
    def set_params(self,**parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
	

