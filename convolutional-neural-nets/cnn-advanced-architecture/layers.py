import numpy as np
import tensorflow as tf

class TerminalLayer(object):
    #final activation layer 
    #i = layer depth/layer-id
    #a = activation function
    #mi = number of input features
    #mo = number of output features    
    def __init__(self, param):
        self.architecture = param
        self.layer_type = param['type']
        self.activation_func = param['activation']
        self.layer_depth = param['depth']
        mi = param['num_input']
        mo = param['num_output']
        tag = self.layer_type + str(self.layer_depth)
        self.W = tf.Variable(tf.truncated_normal([mi, mo], stddev=0.1), name="W"+tag)
        self.b = tf.Variable(tf.constant(0.1, shape=[mo]), name="b"+tag)

    def printArchitecture(self):
        print(self.architecture)
        
    def forward(self,X):
        out = tf.matmul(X, self.W) + self.b
        if self.activation_func == "softmax":
            out = tf.nn.softmax(out)
        #elif self.activation_func == "gap":
        #  return tf.nn.tanh(out)
        return out
        
"""
class DOLayer(object):
    #Drop Out
    #_i = layer depth/layer-id
    #r = Drop Out rate
    def __init__(self, param):
self.architecture = param
        self.layer_type = param['type']
        self.layer_depth = param['depth']
        self.drop_out = param['drop_out']/100.0
        
    def forward(self,X):
        return tf.nn.dropout(X, rate=self.drop_out)
"""

class FCLayer(object):
    #Fully Connected layer
    #_i = layer depth/layer-id
    #mi = number of input features
    #mo = number of output features
    #a = activation function
    def __init__(self, param):
        self.architecture = param
        self.layer_type = param['type']
        self.activation_func = param['activation']
        self.layer_depth = param['depth']
        self.drop_out = param['drop_out']/100.0
        mi = param['num_input']
        mo = param['num_output']
        tag = self.layer_type + str(self.layer_depth)
        self.W = tf.Variable(tf.truncated_normal([mi, mo], stddev=0.1),name="W"+tag)
        self.b = tf.Variable(tf.constant(0.1, shape=[mo]), name="b"+tag)
        
    def printArchitecture(self):
        print(self.architecture)

    def forward(self, X):
        out = tf.matmul(X, self.W) + self.b
        if self.activation_func == "relu":
            out = tf.nn.relu(out)
        elif self.activation_func == "tanh":
            out = tf.nn.tanh(out)
        elif self.activation_func == "sigmoid":
            out = tf.nn.sigmoid(out)
        return tf.nn.dropout(out, rate=self.drop_out)

class ConvLayer(object):
    #Convolution Layer
    #_i = layer depth
    #mi = num_input features
    #mo = num_output_features
    #ksize = convolution kernel_size
    #l = strides
    #a = activation function    
    def __init__(self, param):
        self.architecture = param
        self.layer_type = param['type']
        self.layer_depth = param['depth']
        self.ksize = param['kernel_size']
        self.stride = param['stride']
        self.drop_out = param['drop_out']/100.0
        self.activation_func = param['activation']
        mi = param['num_input']
        mo = param['num_output']
        fw, fh = self.ksize
        tag = self.layer_type + str(self.layer_depth)
        self.W = tf.Variable(tf.truncated_normal([fw, fh, mi, mo], stddev=0.03), name="W"+tag)
        self.b = tf.Variable(tf.constant(0.1, shape=[mo]), name="b"+tag)

    def printArchitecture(self):
        print(self.architecture)

    def forward(self, X):
        sw, sh = self.stride
        conv_out = tf.nn.conv2d(X, self.W, strides=[1, sw, sh, 1], padding='SAME')
        conv_out = tf.nn.bias_add(conv_out, self.b)
        if self.activation_func == "relu":
            conv_out = tf.nn.relu(conv_out)
        elif self.activation_func == "tanh":
            conv_out = tf.nn.tanh(conv_out)
        elif self.activation_func == "sigmoid":
            conv_out = tf.nn.sigmoid(conv_out)
        return tf.nn.dropout(conv_out, rate=self.drop_out)
    
    #f = receptive field size
    #d = num_receptive_fields
    #n = representation size
    def getRepresentations(self):
        #TODO
        return (0,0,0)                     
    
    
class PoolLayer(object):
    #Pooling Layer
    #ksize = pooling kernel_size
    #l = strides
    def __init__(self, param):
        self.architecture = param
        self.layer_type = param['type']
        self.pool = param['pool']        
        self.layer_depth = param['depth']
        self.stride = param['stride']
        self.ksize = param['kernel_size']
        self.drop_out = param['drop_out']/100.0
        
    def printArchitecture(self):
        print(self.architecture)

    def forward(self, X):
        sw, sh = self.stride
        fw, fh = self.ksize
        pool_out = None
        if self.pool == 'max':
            pool_out = tf.nn.max_pool(
                X,
                ksize=[1, fw, fh, 1],
                strides=[1, sw, sh, 1],
                padding='SAME'
            )
        elif self.pool == 'avg':
            pool_out = tf.nn.avg_pool(
                X,
                ksize=[1, fw, fh, 1],
                strides=[1, sw, sh, 1],
                padding='SAME'
            )
        return tf.nn.dropout(pool_out, rate=self.drop_out)
