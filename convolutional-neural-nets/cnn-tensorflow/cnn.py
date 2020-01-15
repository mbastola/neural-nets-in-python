"""
Manil Bastola
Class Implementation of CNN in tensorflow with Sklearn's fit-predict template.
"""

from utils import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle

class FCLayer(object):
    def __init__(self, M1, M2, _id):
        self.id = _id
        self.M1 = M1
        self.M2 = M2
        W, b = init_wb(M1, M2)
        self.W = tf.Variable(W.astype(np.float32)) 
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W, self.b]   
        
    def forward(self, X):
        return tf.nn.relu(tf.matmul(X, self.W) + self.b)
    
class ConvPoolLayer(object):
    #mi = number of input feature maps
    #m0 = number of output feature maps
    #fw = width
    #fh = height
    #poolsize
    def __init__(self, mi, mo, fw = 5, fh=5, poolsize = (2,2)):
        sz = (fw, fh, mi, mo)
        W0 = init_filter(sz, poolsize)
        self.W = tf.Variable(W0)
        b0 = np.zeros(mo,dtype=np.float32)
        self.b = tf.Variable(b0)
        self.poolsize = poolsize
        self.params = [self.W, self.b]
        return
    
    def forward(self, X):
        conv_out = tf.nn.conv2d(X, self.W, strides=[1, 1, 1, 1], padding='SAME')
        conv_out = tf.nn.bias_add(conv_out, self.b)
        p1, p2 = self.poolsize
        pool_out = tf.nn.max_pool(
            conv_out,
            ksize=[1, p1, p2, 1],
            strides=[1, p1, p2, 1],
            padding='SAME'
        )
        return tf.tanh(pool_out)
    
class CNN(object):
    def __init__(self, name, convpool_layer_sizes = [(20,5,5),(20,5,5)], fc_layer_sizes = [30], dropout_rates = None, lr = 1e-3, mu = 0.99, reg = 1e-3, decay = 0.99999, eps =  1e-10 ):
        #architecture_params: [convpool_layer_sizes, fc_layer_sizes, dropout_rates(nullable)] 
        #architecture_params are immutable once the object is constructed

        self.name = name
        self.convpool_layer_sizes = convpool_layer_sizes #pool_layer_sizes
        self.fc_layer_sizes = fc_layer_sizes
        self.dropout_rates =  dropout_rates
        self.drop_out_regularizer = self.dropout_rates != None
        self.K = 2
        
        self.convpool_layers = []
        self.fc_layers = []
        
        #hyper_params: [learning_rate, momentum, regularizer, decay_rate, epsilon]

        self.lr = lr
        self.mu = mu
        self.reg = reg
        self.decay = decay
        self.eps = eps
        
    def fit(self, X, Y, batch_sz=500, epochs=1, method='adam',debug=False):
        lr = np.float32(self.lr)
        mu = np.float32(self.mu)
        reg = np.float32(self.reg)
        decay = np.float32(self.decay)
        eps = np.float32(self.eps)
        self.K = len(set(Y))
        
        # make a validation set
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Y = oneHotEncoder(Y).astype(np.float32)
        
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]
        Yvalid_flat = np.argmax(Yvalid, axis=1) # for calculating error rate
        # initialize convpool layers
        N, width, height,c = X.shape

        mi = c
        outw = width
        outh = height
   
        for mo, fw, fh in self.convpool_layer_sizes:
            layer = ConvPoolLayer(mi, mo, fw, fh)
            self.convpool_layers.append(layer)
            outw = outw // 2
            outh = outh // 2
            mi = mo

        # initialize mlp layers
        M1 = self.convpool_layer_sizes[-1][0]*outw*outh # size must be same as output of last convpool layer
        count = 0
        for M2 in self.fc_layer_sizes:
            h = FCLayer(M1, M2, count)
            self.fc_layers.append(h)
            M1 = M2
            count += 1

        # logistic regression layer
        W, b = init_wb(M1, self.K)
        self.W = tf.Variable(W, 'W_logreg')
        self.b = tf.Variable(b, 'b_logreg')
        
        # collect params for later use
        self.params = [self.W, self.b]
        
        for h in self.convpool_layers:
            self.params += h.params
        for h in self.fc_layers:
            self.params += h.params

        # set up tensorflow functions and variables
        tfX = tf.placeholder(tf.float32, shape=(None, width, height, c), name='X')
        tfY = tf.placeholder(tf.float32, shape=(None, self.K), name='Y')
        act = self.forward(tfX,self.drop_out_regularizer)

        rloss = reg*sum([tf.nn.l2_loss(p) for p in self.params])
    
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=act,
                labels=tfY
            )
        ) + rloss
        prediction = self.predict(tfX)
        
        if (method=="rms"):
            train_op = tf.train.RMSPropOptimizer(lr, decay=decay, momentum=mu).minimize(loss)
        elif (method=="momentum"):
            train_op = tf.train.MomentumOptimizer(lr, momentum=mu).minimize(loss)
        elif (method=="adam"):
            train_op = tf.train.AdamOptimizer(lr).minimize(loss)
        else:
            print("gd loss minimizer unkown")
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

                        p = session.run(prediction, feed_dict={tfX: Xvalid, tfY: Yvalid})
                        e = error_rate(Yvalid_flat, p)
                        train_errors.append(e)
                        if debug:
                            print("i:", i, "j:", j, "nb:", "loss:", l, "error rate:", e)
            model_saver.save(session, "models/"+self.name)
                        
        if debug:
            fig, axes = plt.subplots(nrows=1,ncols=2) 
            axes[0].plot(losses)
            axes[0].set_title('gd loss vs epoch')
            axes[1].plot(train_errors)
            axes[1].set_title('training error vs epoch')
            plt.tight_layout()
            plt.savefig("imgs/"+self.name+".png")
        print ('\nerror_rate: ', train_errors[-1], ', params: ', {'lr':self.lr,'mu':self.mu,'reg':self.reg,'decay': self.decay,'eps':self.eps})

    def forward(self, X, drop_out=False):
        Z = X
        if drop_out:
            Z = tf.nn.dropout(Z, self.dropout_rates[0])
        for c in self.convpool_layers:
            Z = c.forward(Z)
        Z_shape = Z.get_shape().as_list()
        Z = tf.reshape(Z, [-1, np.prod(Z_shape[1:])])
        if drop_out:
            for h, p in zip(self.fc_layers, self.dropout_rates[1:]):
                Z = h.forward(Z)
                Z = tf.nn.dropout(Z, p)
        else:
            for h in self.fc_layers:
                Z = h.forward(Z)
        return tf.matmul(Z, self.W) + self.b

    def predict(self, X):
        if (str(type(X)) == "<class 'numpy.ndarray'>"):
            #init = tf.global_variables_initializer()
            with tf.Session() as session:
                #session.run(init)
                load_model = tf.train.import_meta_graph("models/"+self.name+'.meta')
                load_model.restore(session, tf.train.latest_checkpoint('./models/'))
                                                
                N, width, height,c = X.shape
                tfX = tf.placeholder(tf.float32, shape=(None, width, height, c), name='X')
                #Y = tf.placeholder(tf.float32, shape=(None, self.K), name='Y')
                pY = self.forward(tfX)
                prediction = tf.argmax(pY, 1)
                p = session.run(prediction, feed_dict={tfX: X})
                return p
        else:
            pY = self.forward(X,self.drop_out_regularizer)
            return tf.argmax(pY, 1)
    
    def score(self, X,y):
        pred = self.predict(X)
        return np.mean(y == pred)
