from utils import *
from layers import *
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle


class CNN(object):
    def __init__(self, name, architecture_params, learning_params={'lr':1e-3, 'mu':0.99, 'reg':1e-3, 'decay':0.99999, 'eps':1e-10}):
        #architecture_params: [C,C,P,C,P,FC,FC,T]
        #learning_params: [learning_rate, momentum, regularizer, decay_rate, epsilon]
        self.name = "CNN_" +name + "_" + uniqueStr();
        self.architecture_params = architecture_params
        self.lr = np.float32(learning_params['lr'])
        self.mu = np.float32(learning_params['mu'])
        self.reg = np.float32(learning_params['reg'])
        self.decay = np.float32(learning_params['decay'])
        self.eps = np.float32(learning_params['eps'])
        self.layers = []
        self.K = 2

    def build(self, initial_shape, debug):
        N, width, height, ch = initial_shape
        mi = ch
        outw = width
        outh = height
        i = 0
        cp_first = True
        fc_first = True
        for layer_param in self.architecture_params:
            layer = None
            layer_type = layer_param['type']
            layer_param['depth'] = i
            if layer_type == 'C':
                outw =math.ceil(outw/layer_param['stride'][0])
                outh =math.ceil(outh/layer_param['stride'][0])
                layer_param['num_input'] = mi
                layer = ConvLayer(layer_param)
                cp_first = False
                mi = layer_param['num_output']
            elif layer_type == 'P':
                outw =math.ceil(outw/layer_param['stride'][0])
                outh =math.ceil(outh/layer_param['stride'][0])
                layer =  PoolLayer(layer_param)
            elif layer_type == 'DO':
                layer = DOLayer(layer_param)
            elif layer_type == 'FC':
                if fc_first:
                    # size must be flattened output of last convpool layer
                    mi *= outw * outh
                    fc_first = False
                layer_param['num_input'] = mi
                layer = FCLayer(layer_param)
                mi = layer_param['num_output']
            elif layer_type == 'T':
                layer_param['num_input'] = mi
                layer_param['num_output'] = self.K
                layer = TerminalLayer(layer_param)
            self.layers.append(layer)
            i+=1;

        if debug:
            self.printArchitecture()
        print("built CNN")
            
    def printArchitecture(self):
        for layer in self.layers:
            layer.printArchitecture()
        
    def forward(self, X):
        in_ = X
        out_ = None
        fc_first = True
        for layer in self.layers:
            layer_type = layer.layer_type
            if (layer_type == 'FC') and (fc_first):
                #flatten before dense layer
                in_shape = in_.get_shape().as_list()
                in_ = tf.reshape(in_, [-1, np.prod(in_shape[1:])])
                fc_first = False
            out_ = layer.forward(in_)
            in_ = out_
        return out_

    def computeLoss(self, X,y):
        non_terminal_layers = self.layers[:-1]
        terminal_layer = self.layers[-1]
        params = [terminal_layer.W, terminal_layer.b]
        for layer in non_terminal_layers:
            if (layer.layer_type == 'C') or (layer.layer_type == 'FC'):
                params += [layer.W, layer.b]

        regularized_loss = self.reg*sum([tf.nn.l2_loss(p) for p in params])

        y_pred = self.forward(X)
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,labels=y))
        loss = cross_entropy_loss + regularized_loss 
        return loss

    def fit(self, X, Y, batch_sz=500, epochs=1, method="adam",debug=False):
        self.K = len(set(Y))
        
        # make a validation set
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Y = oneHotEncoder(Y).astype(np.float32)
        
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]
        Yvalid_flat = np.argmax(Yvalid, axis=1) # for calculating error rate
        
        # initialize the architecture
        self.params = self.build(X.shape, debug)
        N, width, height,ch = X.shape
        
        # set up tensorflow functions and variables
        tfX = tf.placeholder(tf.float32, shape=(None, width, height, ch), name='X')
        tfY = tf.placeholder(tf.float32, shape=(None, self.K), name='Y')
        
        loss = self.computeLoss(tfX, tfY)
        prediction = self.predict(tfX)
                
        if (method=="rms"):
            train_op = tf.train.RMSPropOptimizer(self.lr, decay=self.decay, momentum=self.mu).minimize(loss)
        elif (method=="momentum"):
            train_op = tf.train.MomentumOptimizer(self.lr, momentum=self.mu).minimize(loss)
        elif (method=="adam"):
            train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        elif (method=="gd"):
            train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(loss)
        else:
            print("optimizer unkown")
        n_batches = N // batch_sz
        losses = []
        train_errors = []

        model_saver = tf.train.Saver()
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
                        l = session.run(loss, feed_dict={tfX: Xvalid, tfY: Yvalid})
                        losses.append(l)

                        p = session.run(prediction, feed_dict={tfX: Xvalid})
                        print(set(p), len(set(p)))
                        
                        e = error_rate(Yvalid_flat, p)
                        train_errors.append(e)
                        if debug:
                            print("i:", i, "j:", j, "nb:", "loss:", l, "error rate:", e)
            model_saver.save(session, self.name)
        if debug:
            fig, axes = plt.subplots(nrows=1,ncols=2) 
            axes[0].plot(losses)
            axes[1].plot(train_errors)
            plt.tight_layout()
            plt.show()
        print ('\n training error_rate: ', train_errors[-1])

    def predict(self, X):
        if (str(type(X)) == "<class 'numpy.ndarray'>"):
            uninit = False
            #running previously fit model
            if len(self.layers) == 0:
                uninit = True
                self.params = self.build(X.shape, False)
                
            with tf.Session() as session:
                if uninit:
                    session.run(tf.global_variables_initializer())
                    uninit = False
                load_model = tf.train.import_meta_graph(self.name+'.meta')
                load_model.restore(session, tf.train.latest_checkpoint('./'))
                
                N, width, height,c = X.shape
                tfX = tf.placeholder(tf.float32, shape=(None, width, height, c), name='X')
                #Y = tf.placeholder(tf.float32, shape=(None, self.K), name='Y')
                pY = self.forward(tfX)
                prediction = tf.argmax(pY, 1)
                p = session.run(prediction, feed_dict={tfX: X})
                print(set(p), len(set(p)))
                return p
        else:
            pY = self.forward(X)
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
	

