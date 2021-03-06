import time
import pickle
import numpy as np
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.gradient import disconnected_grad as dg
import lasagne

from model import Model

from layers import BernoulliSampleLayer
from distributions import log_bernoulli
from helpers import Tlogsumexp

from theano.tensor.shared_randomstreams import RandomStreams

# ----------------------------------------------------------------------------

class SBN(Model):
    """Sigmoid Belief Network trained using Neural Variational Inference
       Epoch 200 of 200 took 26.052s (192 minibatches)
            training loss/acc:        125.989901  107.652437
            validation loss/acc:      126.220432  108.006230
    """

    def __init__(
        self, n_dim, n_out, n_chan=1, n_superbatch=12800,
        opt_alg='adam', opt_params={'lr': 1e-3, 'b1': 0.9, 'b2': 0.99}
    ):
        # invoke parent constructor
        Model.__init__(self, n_dim, n_chan, n_out, n_superbatch, opt_alg, opt_params)

        # random number generators
        self.numpy_rng = np.random.RandomState(1234)
        self.theano_rng = RandomStreams(self.numpy_rng.randint(2 ** 30))

        (loss, acc) = self.objectives
        llik = self.create_llik()
        (X, Y, idx1, idx2, S) = self.inputs
        # self.llik = theano.function(
        #     [idx1, idx2], llik, updates=self.updates,
        #     {X: self.train_set_x[idx1:idx2], Y: self.train_set_y[idx1:idx2]}
        # )
        self.loss = theano.function(
            [X, Y], [loss, llik],
            on_unused_input='warn',
        )

    def create_model(self, X, Y, n_dim, n_out, n_chan=1):
        # params
        n_lat    = 200 # latent stochastic variables
        n_hid    = 500 # size of hidden layer in encoder/decoder
        n_hid_cv = 500 # size of hidden layer in control variate net
        n_out    = n_dim * n_dim * n_chan # total dimensionality of ouput
        hid_nl   = lasagne.nonlinearities.tanh

        # create the encoder network
        l_q_in = lasagne.layers.InputLayer(
            shape=(None, n_chan, n_dim, n_dim),
            input_var=X,
        )
        l_q_hid = lasagne.layers.DenseLayer(
            l_q_in, num_units=n_hid,
            nonlinearity=hid_nl,
        )
        l_q_out = lasagne.layers.DenseLayer(
            l_q_hid, num_units=n_lat,
            nonlinearity=None,
        )
        l_q_mu = lasagne.layers.DenseLayer(
            l_q_hid, num_units=n_lat,
            nonlinearity=T.nnet.sigmoid,
        )
        l_q_sample = BernoulliSampleLayer(l_q_mu)

        # create the decoder network
        # note that we currently only handle Bernoulli x variables
        l_p_in = lasagne.layers.InputLayer((None, n_lat))
        l_p_hid = lasagne.layers.DenseLayer(
            l_p_in, num_units=n_hid,
            nonlinearity=hid_nl,
            W=lasagne.init.GlorotUniform(),
        )
        l_p_mu = lasagne.layers.DenseLayer(l_p_hid, num_units=n_out,
            nonlinearity = lasagne.nonlinearities.sigmoid,
            W=lasagne.init.GlorotUniform(),
            b=lasagne.init.Constant(0.),
        )

        # create control variate (baseline) network
        l_cv_in = lasagne.layers.InputLayer(
            shape=(None, n_chan, n_dim, n_dim),
            input_var=X,
        )
        l_cv_hid = lasagne.layers.DenseLayer(
            l_cv_in, num_units=n_hid_cv,
            nonlinearity=hid_nl,
        )
        l_cv = lasagne.layers.DenseLayer(
            l_cv_hid, num_units=1,
            nonlinearity=None,
        )

        # create variables for centering signal
        c = theano.shared(np.zeros((1,1), dtype=np.float64), broadcastable=(True,True))
        v = theano.shared(np.zeros((1,1), dtype=np.float64), broadcastable=(True,True))

        self.input_layers = (l_q_in, l_p_in, l_cv_in)
        self.n_lat = n_lat
        self.n_hid = n_hid

        return l_p_mu, l_q_mu, l_q_sample, l_cv, c, v

    def _create_components(self, deterministic=False):
        # load network input
        X = self.inputs[0]
        x = X.flatten(2)

        # load networks
        l_p_mu, l_q_mu, l_q_sample, _, _, _ = self.network
        l_q_in, l_p_in, l_cv_in = self.input_layers

        # load network output
        z, q_mu = lasagne.layers.get_output(
            [l_q_sample, l_q_mu], deterministic=deterministic)
        p_mu = lasagne.layers.get_output(
            l_p_mu, {l_p_in: z},
            deterministic=deterministic,
        )

        # entropy term
        log_qz_given_x = log_bernoulli(dg(z), q_mu).sum(axis=1)

        # expected p(x,z) term
        z_prior = T.ones_like(z)*np.float32(0.5)
        log_pz = log_bernoulli(z, z_prior).sum(axis=1)
        log_px_given_z = log_bernoulli(x, p_mu).sum(axis=1)
        log_pxz = log_pz + log_px_given_z

        # save them for later
        self.log_pxz = log_pxz
        self.log_qz_given_x = log_qz_given_x

        return log_pxz.flatten(), log_qz_given_x.flatten()

    def create_objectives(self, deterministic=False):
        # load probabilities
        log_pxz, log_qz_given_x = self._create_components(deterministic=deterministic)

        # compute the lower bound
        elbo = T.mean(log_pxz - log_qz_given_x)

        # we don't use the second accuracy metric right now
        return -elbo, -T.mean(log_qz_given_x)

    def create_gradients(self, loss, deterministic=False):
        from theano.gradient import disconnected_grad as dg

        # load networks
        l_p_mu, l_q_mu, _, l_cv, c, v = self.network

        # load params
        p_params  = lasagne.layers.get_all_params(l_p_mu, trainable=True)
        q_params  = lasagne.layers.get_all_params(l_q_mu, trainable=True)
        cv_params = lasagne.layers.get_all_params(l_cv, trainable=True)

        # load neural net outputs (probabilities have been precomputed)
        log_pxz, log_qz_given_x = self.log_pxz, self.log_qz_given_x
        cv = T.addbroadcast(lasagne.layers.get_output(l_cv),1)

        # compute learning signals
        l = log_pxz - log_qz_given_x - cv
        l_avg, l_var = l.mean(), l.var()
        c_new = 0.8*c + 0.2*l_avg
        v_new = 0.8*v + 0.2*l_var
        l = (l - c_new) / T.maximum(1, T.sqrt(v_new))

        # compute grad wrt p
        p_grads = T.grad(-log_pxz.mean(), p_params)

        # compute grad wrt q
        q_target = T.mean(dg(l) * log_qz_given_x)
        q_grads = T.grad(-0.2*q_target, q_params) # 5x slower rate for q

        # compute grad of cv net
        cv_target = T.mean(l**2)
        cv_grads = T.grad(cv_target, cv_params)

        # combine and clip gradients
        clip_grad = 1
        max_norm = 5
        grads = p_grads + q_grads + cv_grads
        mgrads = lasagne.updates.total_norm_constraint(grads, max_norm=max_norm)
        cgrads = [T.clip(g, -clip_grad, clip_grad) for g in mgrads]

        return cgrads

    def gen_samples(self, deterministic=False):
        s = self.inputs[-1]
        # put it through the decoder
        _, l_p_in, _ = self.input_layers
        l_p_mu = self.network[0]
        p_mu = lasagne.layers.get_output(l_p_mu, {l_p_in : s})

        return p_mu

    def get_params(self):
        l_p_mu, l_q_mu, _, l_cv, _, _ = self.network
        p_params  = lasagne.layers.get_all_params(l_p_mu, trainable=True)
        q_params  = lasagne.layers.get_all_params(l_q_mu, trainable=True)
        cv_params = lasagne.layers.get_all_params(l_cv, trainable=True)
        return p_params + q_params + cv_params #+ [c]

    def create_updates(self, grads, params, alpha, opt_alg, opt_params):
        # call super-class to generate SGD/ADAM updates
        grad_updates = Model.create_updates(
            self, grads, params, alpha, opt_alg, opt_params,
        )

        # create updates for centering signal

        # load neural net outputs (probabilities have been precomputed)
        _, _, _, l_cv, c, v = self.network
        log_pxz, log_qz_given_x = self.log_pxz, self.log_qz_given_x
        cv = T.addbroadcast(lasagne.layers.get_output(l_cv),1)

        # compute learning signals
        l = log_pxz - log_qz_given_x - cv
        l_avg, l_var = l.mean(), l.var()
        c_new = 0.8*c + 0.2*l_avg
        v_new = 0.8*v + 0.2*l_var

        # compute update for centering signal
        cv_updates = {c: c_new, v: v_new}

        return OrderedDict(grad_updates.items() + cv_updates.items())

    def create_llik(self):
        # load inputs
        X = self.inputs[0]
        x = X.flatten(2)

        # load network params
        n_cat = self.n_lat
        n_rep = 10

        # load networks
        l_p_mu, l_q_mu, l_q_sample, _, _, _ = self.network
        l_q_in, l_p_in, l_cv_in = self.input_layers

        # load network output
        q_mu = lasagne.layers.get_output(l_q_mu)

        q_mu_rep = T.tile( q_mu.dimshuffle((0,'x',1)), reps=(1,n_rep,1) ) # (n_bat, n_rep, n_cat)
        q_sample_hard = self.theano_rng.binomial(size=q_mu_rep.shape, p=q_mu_rep, dtype=q_mu_rep.dtype) # (n_bat, n_rep, n_cat)
        q_sample_hard2 = q_sample_hard.reshape([100*n_rep, n_cat]) # (n_bat*n_rep, n_cat)

        p_mu = lasagne.layers.get_output(l_p_mu, {l_p_in: q_sample_hard2}) # (n_bat*n_rep, 784)
        x_rep = T.tile( x.dimshuffle((0,'x',1)), reps=(1,n_rep,1) ) # (n_bat, n_rep, 784)
        p_mu = T.reshape(p_mu, (100, n_rep, 784)) # (n_bat, n_rep, 784)

        # define the loss components
        log_p_x = log_bernoulli(x_rep, p_mu).sum(axis=2) # (n_bat, n_rep)
        z_prior = T.ones_like(q_sample_hard)*np.float32(0.5) # (n_bat, n_rep, n_cat)
        log_p_z = log_bernoulli(q_sample_hard, z_prior).sum(axis=2) # (n_bat, n_rep)
        log_q_z = log_bernoulli(q_sample_hard, q_mu_rep).sum(axis=2) # (n_bat, n_rep)

        # compute loss
        llik = Tlogsumexp( log_p_x + log_p_z - log_q_z, axis=1) # (n_bat,)

        return T.mean(llik)
