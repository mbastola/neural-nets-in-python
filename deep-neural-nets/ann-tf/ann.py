from utils import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle

class ANNLayer(object):
    def __init__(self, _id, M1, M2, drop_out = 0, activation="relu"):
        self.id = _id
        self.M1 = M1
        self.M2 = M2
        self.activation_func = activation
        self.drop_out = drop_out
        self.W = tf.Variable(tf.truncated_normal([M1, M2], stddev=0.1),name = "W"+str(_id))
        self.b = tf.Variable(tf.constant(0.1, shape=[M2]), name="b"+str(_id))
        self.params = [self.W, self.b]   
        
    def forward(self, X):
        out = tf.matmul(X, self.W) + self.b
        if self.activation_func == "relu":
            out = tf.nn.relu(out)
        elif self.activation_func == "tanh":
            out = tf.nn.tanh(out)
        elif self.activation_func == "sigmoid":
            out = tf.nn.sigmoid(out)
        return tf.nn.dropout(out, rate=self.drop_out)

class ANN(object):
    def __init__(self, name, layer_sizes, dropout_rates = None, lr = 1e-3, mu = 0.99, reg = 1e-3, decay = 0.99999, eps =  1e-10 ):
        self.name = "ANN_" + name #+ "_" + uniqueStr();
        self.layer_sizes = layer_sizes
        self.dropout_rates =  dropout_rates
        if dropout_rates == None:
            self.dropout_rates = np.zeros(len(layer_sizes))
            
        self.K = 2
        
        self.layers = []
        
        self.lr = lr
        self.mu = mu
        self.reg = reg
        self.decay = decay
        self.eps = eps

    def forward(self, X):
        Z = X
        for h in self.layers:
            Z = h.forward(Z)
        #tf.matmul(Z, self.W) + self.b
        return Z  

    def computeLoss(self, X,y):
        non_terminal_layers = self.layers[:-1]
        terminal_layer = self.layers[-1]
        params = [terminal_layer.W, terminal_layer.b]
        for layer in non_terminal_layers:
            params += [layer.W, layer.b]

        regularized_loss = self.reg*sum([tf.nn.l2_loss(p) for p in params])

        y_pred = self.forward(X)
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,labels=y))
        loss = cross_entropy_loss + regularized_loss 
        return loss

    def build(self, initial_shape, debug):
        idx = 0
        N, d = initial_shape
        M1 = d
        for layer_size in self.layer_sizes:
            M2 = layer_size
            print(M1,M2, self.dropout_rates[idx])
            layer = ANNLayer(idx, M1,M2, self.dropout_rates[idx])
            self.layers.append(layer)
            M1 = M2
            idx+=1
        if debug:
            print(self.layers)
        return
            
    def fit(self, X, Y, batch_sz=500, epochs=1, method="adam",debug=False):
        self.K = len(set(Y))
        
        # make a validation set
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Y = oneHotEncoder(Y).astype(np.float32)
        
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]
        Yvalid_flat = np.argmax(Yvalid, axis=1) #oneHotDecode
        
        # initialize the architecture
        self.params = self.build(X.shape, debug)
        N, f = X.shape
        
        # set up tensorflow functions and variables
        tfX = tf.placeholder(tf.float32, shape=(None, f), name='X')
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

                    #if j == (n_batches-1):
                    if j == 20:
                        l = session.run(loss, feed_dict={tfX: Xvalid, tfY: Yvalid})
                        losses.append(l)

                        p = session.run(prediction, feed_dict={tfX: Xvalid})
                        print(set(p), len(set(p)))
                        
                        e = error_rate(Yvalid_flat, p)
                        train_errors.append(e)
                        if debug:
                            print("i:", i, "j:", j, "nb:", "loss:", l, "error rate:", e)
            model_saver.save(session, "models/"+self.name)
        if debug:
            fig, axes = plt.subplots(nrows=1,ncols=2) 
            axes[0].plot(losses)
            axes[1].plot(train_errors)
            plt.tight_layout()
            #plt.show()
            plt.savefig("imgs/"+self.name+".png")
        print ('\n training error_rate: ', train_errors[-1])
    
    def score(self, X,y):
        pred = self.predict(X)
        return np.mean(y == pred)
    
    def predict(self, X):
        if (str(type(X)) == "<class 'numpy.ndarray'>"):
            uninit = False
            #running previously fit model
            if len(self.layers) == 0:
                uninit = True
                print("running previously fit model")
                input()
                self.params = self.build(X.shape, False)
                
            with tf.Session() as session:
                if uninit:
                    session.run(tf.global_variables_initializer())
                    uninit = False
                load_model = tf.train.import_meta_graph("models/"+self.name+'.meta')
                load_model.restore(session, tf.train.latest_checkpoint('./models/'))
                
                N, f = X.shape
                tfX = tf.placeholder(tf.float32, shape=(None, f), name='X')
                #Y = tf.placeholder(tf.float32, shape=(None, self.K), name='Y')
                pY = self.forward(tfX)
                prediction = tf.argmax(pY, 1)
                p = session.run(prediction, feed_dict={tfX: X})
                return p
        else:
            pY = self.forward(X)
            return tf.argmax(pY, 1)
    
