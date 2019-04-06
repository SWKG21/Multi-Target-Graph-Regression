import keras.backend as K
from keras.layers import Layer
from keras import initializers, regularizers, constraints
from keras.layers import Input, Embedding, Dropout, TimeDistributed, Dense, Lambda, Conv2D, GlobalMaxPooling1D, Concatenate

from utils import *


class CNNencoder(Layer):
    """
    initially taken from: https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    
    Note: The layer has been tested with Keras 2.0.6
    
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(CNNencoder())
        # next add a Dense layer (for classification/regression) or whatever...
    """
    
    def __init__(self, nfilters, cnn_windows, embed_dim, return_coefficients=False,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.return_coefficients = return_coefficients
        self.init = initializers.get('glorot_uniform')
        
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)
        
        self.bias = bias

        # self.nfilters = nfilters
        # self.cnn_windows = cnn_windows

        self.conv2ds = []
        self.sqs = []
        self.mps = []
        for window in cnn_windows:
            self.conv2ds.append(Conv2D(filters=nfilters, kernel_size=(window, embed_dim), data_format='channels_first'))  # (batch_size, #filters, sent_len-window+1, 1)
            self.sqs.append(Lambda(lambda x: K.squeeze(x, axis=-1)))  # (batch_size, #filters, sent_len-window+1)
            self.mps.append(GlobalMaxPooling1D(data_format='channels_first'))  # (batch_size, #filters)
        super(CNNencoder, self).__init__(**kwargs)
    
    
    def build(self, input_shape):
        assert len(input_shape) == 3
        super(CNNencoder, self).build(input_shape)
    
    
    def call(self, x, mask=None):
        # x with shape (batch_size, sent_len, embedding_dim)
        x = K.expand_dims(x, axis=1)
        return x
        # x = Lambda(lambda y: K.expand_dims(y, axis=1))(x)  # (batch_size, 1, sent_len, embedding_dim)
        
        # # tmp = self.conv2ds[0](x)
        # # tmp = self.sqs[0](tmp)
        # # tmp = self.mps[0](tmp)
        # # return tmp

        # # convs = []
        # # for window in self.cnn_windows:
        # #     tmp = Conv2D(filters=self.nfilters, strides=(1, 1), padding='valid', dilation_rate=(1, 1), kernel_size=(window, K.int_shape(x)[-1]), data_format='channels_first')(x)  # (batch_size, #filters, sent_len-window+1, 1)
        # #     tmp = Lambda(lambda x: K.squeeze(x, axis=-1))(tmp)  # (batch_size, #filters, sent_len-window+1)
        # #     tmp = GlobalMaxPooling1D(data_format='channels_first')(tmp)  # (batch_size, #filters)
        # #     convs.append(tmp)
        
        # convs = []
        # for i in range(len(self.conv2ds)):
        #     tmp = self.conv2ds[i](x)
        #     tmp = self.sqs[i](tmp)
        #     tmp = self.mps[i](tmp)
        #     convs.append(tmp)

        # output = Concatenate(axis=1)(convs)  # (batch_size, #filters*#windows)
        # return output
    
    
    def compute_output_shape(self, input_shape):
        if self.return_coefficients:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[-1], 1)]
        else:
            return input_shape[0], input_shape[-1]

