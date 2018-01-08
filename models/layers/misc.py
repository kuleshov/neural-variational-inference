import numpy as np
import theano
import theano.tensor as T
import lasagne

# ----------------------------------------------------------------------------

class ArgmaxLayer(lasagne.layers.Layer):
    """Argmax along the last dimension"""
    def __init__(self, incoming, **kwargs):
        super(ArgmaxLayer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        return input_shape[:-1]

    def get_output_for(self, input, **kwargs):
        return T.argmax(input, axis=-1)

class RoundLayer(lasagne.layers.Layer):
    """Argmax along the last dimension"""
    def __init__(self, incoming, **kwargs):
        super(RoundLayer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        x = T.sgn(input)
        x = .5*(x + 1.)
        return x

class SoftmaxLayer(lasagne.layers.Layer):
    """Argmax along the last dimension"""
    def __init__(self, incoming, tau=1., **kwargs):
        super(SoftmaxLayer, self).__init__(incoming, **kwargs)
        self.tau = tau

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        return T.nnet.softmax(input / self.tau)

class SigmoidLayer(lasagne.layers.Layer):
    """Argmax along the last dimension"""
    def __init__(self, incoming, tau=1., **kwargs):
        super(SigmoidLayer, self).__init__(incoming, **kwargs)
        self.tau = tau

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        return T.nnet.sigmoid(input / self.tau)        