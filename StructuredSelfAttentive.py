import keras.backend as K
from keras.layers import Layer
from keras import initializers, regularizers, constraints

from utils import *
    

class StructuredSelfAttentive(Layer):
    """
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(StructuredSelfAttentive())
        # next add a Dense layer (for classification/regression) or whatever...
    """
    
    def __init__(self, da, r, return_coefficients=False,
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
    

    def call(self, x, mask=None):
        ait = dot_product(K.tanh(dot_product(x, self.W_s1)), self.W_s2)
        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        
        a = K.permute_dimensions(a, (0, 2, 1))
        weighted_input = K.batch_dot(a, x)

        if self.return_coefficients:
            return [K.sum(weighted_input, axis=1), a]
        else:
            return K.sum(weighted_input, axis=1)
    
    
    def compute_output_shape(self, input_shape):
        if self.return_coefficients:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[-1], 1)]
        else:
            return input_shape[0], input_shape[-1]

