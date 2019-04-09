import keras.backend as K
from keras.layers import Layer
from keras import initializers, regularizers, constraints

from utils import *


class AttentionWithContext(Layer):
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
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """
    
    def __init__(self, return_coefficients=False,
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
        super(AttentionWithContext, self).__init__(**kwargs)
    
    
    def build(self, input_shape):
        assert len(input_shape) == 3
        
        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        
        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)
        
        super(AttentionWithContext, self).build(input_shape)
    
    
    def call(self, x, mask=None):
        uit = dot_product(x, self.W)
        
        if self.bias:
            uit += self.b
        
        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)
        
        a = K.exp(ait)
        
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        
        a = K.expand_dims(a)
        weighted_input = x * a

        if self.return_coefficients:
            return [K.sum(weighted_input, axis=1), a]
        else:
            return K.sum(weighted_input, axis=1)
    
    
    def compute_output_shape(self, input_shape):
        if self.return_coefficients:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[-1], 1)]
        else:
            return input_shape[0], input_shape[-1]



class SkipConnection(Layer):

    def __init__(self, W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)
        
        self.bias = bias
        super(SkipConnection, self).__init__(**kwargs)
    
    
    def build(self, input_shapes):
        assert len(input_shapes[0]) == 2
        assert len(input_shapes[1]) == 3
        
        self.W = self.add_weight((input_shapes[0][-1], input_shapes[1][-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shapes[0][-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        
        super(SkipConnection, self).build(input_shapes)
    
    
    def call(self, xs):
        transformed = xs[0]  # (batch_size, 2*sent_n_units) / (batch_size, 2*doc_n_units)

        original = xs[1]  # (batch_size, sent_len, embed_dim) / (batch_size, doc_len, 2*sent_n_units)
        original = K.mean(original, axis=1)  # (batch_size, embed_dim) / (batch_size, 2*sent_n_units)
        modif = dot_product(original, self.W)  # (batch_size, 2*sent_n_units) / (batch_size, 2*doc_n_units)
        if self.bias:
            modif += self.b
        
        sc = transformed + modif  # (batch_size, 2*sent_n_units) / (batch_size, 2*doc_n_units)
        return sc



class SelfAttention(AttentionWithContext):
    """
        For sentence encoder, u == mean of word vectors;
        For document encoder, u == mean of sentence vectors
    """
    
    def build(self, input_shape):
        assert len(input_shape) == 3
        
        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

    
    def call(self, x):
        # x with (batch_size, sent_len, 2*sent_n_units) / (batch_size, doc_len, 2*doc_n_units)
        u = K.mean(x, axis=1)  # (batch_size, 2*sent_n_units) / (batch_size, 2*doc_n_units)
        
        uit = dot_product(x, self.W)  # (batch_size, sent_len, 2*sent_n_units) / (batch_size, doc_len, 2*doc_n_units)
        if self.bias:
            uit += self.b
        uit = K.tanh(uit)  # (batch_size, sent_len, 2*sent_n_units) / (batch_size, doc_len, 2*doc_n_units)
        
        # use batch_dot rather dot_product because u shape (?, 2*sent_n_units) rather shape (2*sent_n_units,)
        ait = K.batch_dot(uit, u)  # (batch_size, sent_len) / (batch_size, doc_len)
        a = K.exp(ait)  # (batch_size, sent_len) / (batch_size, doc_len)

        
        # add a very small positive number ε to the sum to avoid NaN's
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())  # (batch_size, sent_len) / (batch_size, doc_len)

        a = K.expand_dims(a)  # (batch_size, sent_len, 1) / (batch_size, doc_len, 1)
        weighted_input = x * a  # (batch_size, sent_len, 2*sent_n_units) / (batch_size, doc_len, 2*doc_n_units)
        
        # sum by sent_len/doc_len, output shape (batch_size, 2*sent_n_units) / (batch_size, 2*doc_n_units)
        if self.return_coefficients:
            return [K.sum(weighted_input, axis=1), a]
        else:
            return K.sum(weighted_input, axis=1)



class StructuredSelfAttentive(Layer):
    """
        For sentence encoder, 
            input: word vectors;
            output: sentence vector
    """
    
    def __init__(self, da, r, return_matrix=False, return_coefficients=False,
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
        
        self.da = da
        self.r = r
        self.return_matrix = return_matrix
        super(StructuredSelfAttentive, self).__init__(**kwargs)
    

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W_s1 = self.add_weight((self.da, input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        self.W_s2 = self.add_weight((self.r, self.da,),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        
        super(StructuredSelfAttentive, self).build(input_shape)
    

    def call(self, x):
        # dimension u in paper == u_n_units here
        # x shape (batch_size, sent_len, 2u)
        ait = dot_product(K.tanh(dot_product(x, self.W_s1)), self.W_s2)  # (batch_size, sent_len, r)
        a = K.exp(ait)  # (batch_size, sent_len, r)
        
        # add a very small positive number ε to the sum to avoid NaN's
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())  # (batch_size, sent_len, r)
        
        a = K.permute_dimensions(a, (0, 2, 1))  # (batch_size, r, sent_len)
        weighted_input = K.batch_dot(a, x)  # (batch_size, r, 2u)
        
        # output a matrix (batch_size, r, 2u) or a vector (batch_size, 2u)
        if self.return_coefficients:
            if self.return_matrix:
                return [weighted_input, a]
            else:
                return [K.mean(weighted_input, axis=1), a]
        else:
            if self.return_matrix:
                return weighted_input
            else:
                return K.mean(weighted_input, axis=1)


    def compute_output_shape(self, input_shape):
        if self.return_coefficients:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[-1], 1)]
        else:
            return input_shape[0], input_shape[-1]



class DocStructuredAttention(AttentionWithContext):
    """
        For document encoder, 
        input: 
            x == sentence vectors;
            u == mean of sentence vectors for u;
        output: 
            document vector
    """
    
    def build(self, input_shapes):
        assert len(input_shapes[0]) == 3
        assert len(input_shapes[1]) == 3
        input_shape = input_shapes[0]
        
        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

    
    def call(self, xs):
        x = xs[0]  # (batch_size, doc_len, 2*doc_n_units)
        uit = dot_product(x, self.W)  # (batch_size, doc_len, 2*doc_n_units)
        if self.bias:
            uit += self.b
        uit = K.tanh(uit)  # (batch_size, doc_len, 2*doc_n_units)
        
        # xs[1] with (batch_size, doc_len, 2*doc_n_units)
        u = K.mean(xs[1], axis=1)  # (batch_size, 2*doc_n_units)
        
        # use batch_dot rather dot_product because u shape (?, 2*doc_n_units) rather (2*doc_n_units,)
        ait = K.batch_dot(uit, u)  # (batch_size, doc_len)
        a = K.exp(ait)  # (batch_size, doc_len)
        
        # add a very small positive number ε to the sum to avoid NaN's
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())  # (batch_size, doc_len)

        a = K.expand_dims(a)  # (batch_size, doc_len, 1)
        weighted_input = x * a  # (batch_size, doc_len, 2*doc_n_units)
        
        # sum by doc_len, output shape (batch_size, 2*doc_n_units)
        if self.return_coefficients:
            return [K.sum(weighted_input, axis=1), a]
        else:
            return K.sum(weighted_input, axis=1)
    
    
    def compute_output_shape(self, input_shapes):
        input_shape = input_shapes[0]
        if self.return_coefficients:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[-1], 1)]
        else:
            return input_shape[0], input_shape[-1]



class SentContextAttention(AttentionWithContext):
    """
        For sentence encoder, 
        input: 
            x == word vectors;
            u == mean of contextual sentence vectors;
        output: 
            sent vector
    """
    
    def build(self, input_shapes):
        assert len(input_shapes[0]) == 3
        
        self.W = self.add_weight((input_shapes[1][-1], input_shapes[0][-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shapes[1][-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

    
    def call(self, xs):
        x = xs[0]  # (batch_size, sent_len, 2*sent_n_units)
        uit = dot_product(x, self.W)  # (batch_size, sent_len, sent_u_dim)
        if self.bias:
            uit += self.b
        uit = K.tanh(uit)  # (batch_size, sent_len, sent_u_dim)
        
        # xs[1] with (batch_size, window_size, sent_u_dim) or (batch_size, sent_u_dim)
        if len(K.int_shape(xs[1])) == 3:
            u = K.mean(xs[1], axis=1)  # (batch_size, sent_u_dim)
        elif len(K.int_shape(xs[1])) == 2:
            u = xs[1]  # (batch_size, sent_u_dim)

        # use batch_dot rather dot_product because u shape (?, sent_u_dim) rather shape (sent_u_dim,)
        ait = K.batch_dot(uit, u)  # (batch_size, sent_len)
        a = K.exp(ait)  # (batch_size, sent_len)
        
        # add a very small positive number ε to the sum to avoid NaN's
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())  # (batch_size, sent_len)
        
        a = K.expand_dims(a)  # (batch_size, sent_len, 1)
        weighted_input = x * a  # (batch_size, sent_len, 2*sent_n_units)
        
        # sum by sent_len, output shape (batch_size, 2*sent_n_units)
        if self.return_coefficients:
            return [K.sum(weighted_input, axis=1), a]
        else:
            return K.sum(weighted_input, axis=1)
    
    
    def compute_output_shape(self, input_shapes):
        input_shape = input_shapes[0]
        if self.return_coefficients:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[-1], 1)]
        else:
            return input_shape[0], input_shape[-1]



class SentMultiContextAttention(AttentionWithContext):
    """
        For sentence encoder, 
        input: 
            x == word vectors;
            u == mean of contextual sentence vectors from AttentionWithContext;
        output: 
            sent vector
    """
    
    def build(self, input_shapes):
        assert len(input_shapes[0]) == 3
        assert len(input_shapes[1]) == 4
        input_shape = input_shapes[0]
        
        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

    
    def call(self, xs, mask=None):
        x = xs[0]  # (batch_size, sent_len, 2*sent_n_units)
        uit = dot_product(x, self.W)  # (batch_size, sent_len, 2*sent_n_units)
        if self.bias:
            uit += self.b
        uit = K.tanh(uit)  # (batch_size, sent_len, 2*sent_n_units)
        
        # xs[1] with (batch_size, window_size, r, 2*sent_n_units)
        u = K.mean(xs[1], axis=1)  # (batch_size, r, 2*sent_n_units)
        u = K.permute_dimensions(u, (0, 2, 1))  # (batch_size, 2*sent_n_units, r)
        
        # use batch_dot rather dot_product because u shape (?, 2*sent_n_units) rather shape (2*sent_n_units,)
        ait = K.batch_dot(uit, u)  # (batch_size, sent_len, r)
        a = K.exp(ait)  # (batch_size, sent_len, r)
        
        # add a very small positive number ε to the sum to avoid NaN's
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())  # (batch_size, sent_len, r)

        a = K.permute_dimensions(a, (0, 2, 1))  # (batch_size, r, sent_len)
        weighted_inputs = K.batch_dot(a, x)  # (batch_size, r, 2*sent_n_units)
        
        # mean by r, output shape (batch_size, 2*sent_n_units)
        if self.return_coefficients:
            return [K.mean(weighted_inputs, axis=1), a]
        else:
            return K.mean(weighted_inputs, axis=1)
    
    
    def compute_output_shape(self, input_shapes):
        input_shape = input_shapes[0]
        if self.return_coefficients:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[-1], 1)]
        else:
            return input_shape[0], input_shape[-1]