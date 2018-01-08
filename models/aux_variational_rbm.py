import pdb
import time
import pickle
import numpy as np
import theano
import lasagne
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict
from lasagne.layers import batch_norm

from model import Model
from abstractrbm import AbstractRBM
from helpers import iterate_minibatch_idx, evaluate, log_metrics
from helpers import Tlogsumexp, Tlogaddexp
from layers import BernoulliSampleLayer, Deconv2DLayer
from distributions import log_bernoulli, log_normal, log_normal2

from theano.compile.nanguardmode import NanGuardMode

create_zmat = lambda x, y: np.zeros((x, y)).astype(theano.config.floatX)
create_vec = lambda y: np.zeros(y,).astype(theano.config.floatX)
create_sca = lambda x: (np.cast[theano.config.floatX](x)).astype(theano.config.floatX)

class AuxiliaryVariationalRBM(AbstractRBM):
    """Restricted Boltzmann Machine with VI"""
    def __init__(
        self, n_dim, n_out, n_chan=1, n_superbatch=12800, opt_alg='adam',
        opt_params={'lr': 1e-3, 'b1': 0.9, 'b2': 0.99}
    ):
        """RBM constructor.
        Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.
        """
        self.numpy_rng = np.random.RandomState(1234)
        self.theano_rng = RandomStreams(self.numpy_rng.randint(2 ** 30))
        self.create_mat = lambda x, y: self.numpy_rng.normal(0, 0.01, (x, y)).astype(theano.config.floatX)

        # save config
        n_batch = opt_params.get('nb')
        self.n_hidden = 8
        self.n_visible = n_chan*n_dim*n_dim  # size of visible layer
        self.n_batch = n_batch
        self.n_mc = 300 # num of monte carlo samples from each MoB component

        self.n_dim = n_dim
        self.n_out = n_out
        self.n_superbatch = n_superbatch
        self.alg = opt_alg

        # set up general RBM methods
        AbstractRBM.__init__(self, n_dim, n_chan, n_out, n_superbatch, opt_alg, opt_params) 

        # create updates
        alpha = T.scalar(dtype=theano.config.floatX)  # learning rate

        # save config
        self.n_class = 2
        self.n_dim = n_dim
        self.n_out = n_out

        self.marginalize = 'h'
        self.n_samples = self.n_mc
        self.n_latent = self.n_visible if self.marginalize == 'h' else self.n_hidden

        # create input variables
        D, idx1, idx2 = self.create_inputs()

        # create model
        self.network = self.create_model()

        # create objectives
        loglik, plik = self.create_objectives(D)

        # create gradients
        dL_Theta, dE_Theta, dlogZ_Theta, dL_qx, dL_pa = self.create_gradients()
        grads = dL_Theta, dE_Theta, dlogZ_Theta, dL_qx, dL_pa

        # create updates
        uL_Theta, uL_Phi, avg_updates, avg_Theta_updates \
          = self.create_updates(grads, None, alpha, opt_alg, opt_params)
      
        # logF_avg, Z_avg = self.create_llik_estimate(D)
        
        mode = NanGuardMode(nan_is_error=True, inf_is_error=False, big_is_error=False)
        mode = None

        common_update1 = OrderedDict(avg_updates.items() + uL_Phi.items())
        self.train_q = theano.function([idx1, idx2], [loglik, self.logbnd], 
          updates=common_update1, mode=mode,
          givens={D: self.train_set_x[idx1:idx2]})
        self.train_q0 = theano.function([D], self.logbnd, updates=common_update1, mode=mode)

        common_update2 = OrderedDict(avg_Theta_updates.items() + uL_Theta.items())
        self.train_p = theano.function([idx1, idx2, alpha], [loglik, self.logbnd], 
            updates=common_update2, mode=mode, on_unused_input='warn',
            givens={D: self.train_set_x[idx1:idx2]})
        self.train_p_batch = theano.function([D, alpha], [loglik, self.logbnd], 
            updates=common_update2, mode=mode, on_unused_input='warn')
        # self.llik = theano.function([D], logF_avg - T.log(Z_avg), mode=mode)

    def create_inputs(self):
        # allocate symbolic variables for the data
        idx1, idx2 = T.lscalar(), T.lscalar()
        D = T.tensor4('D', dtype=theano.config.floatX) # data matrixX)  # learning rate
        # self.Z = T.tensor2(dtype=theano.config.floatX)
        return D, idx1, idx2

    def _free_energy(self, X, marginalize='h', avg=False):
      bv_vec = self.vbias.reshape((self.n_visible,1))
      bh_vec = self.hbias.reshape((self.n_hidden,1))

      W = self.W_avg if avg else self.W
      W = W.T

      if marginalize == 'h':
        logF = T.dot(bv_vec.T, X) \
             + T.sum(T.log(1. + T.exp(bh_vec + T.dot(W, X))), axis=0)
      elif marginalize == 'v':
        logF = T.dot(bh_vec.T, X) \
             + T.sum(T.log(1. + T.exp(bv_vec + T.dot(W.T, X))), axis=0)
      else:
        raise ValueError('invalid argument')
      
      return logF

    @property
    def Z(self):
      return np.exp(self.logZ_avg.get_value())

    def create_model(self):
        # create the RBM
        self.W = theano.shared(self.create_mat(self.n_visible, self.n_hidden))
        self.vbias = theano.shared(create_vec(self.n_visible))
        self.hbias = theano.shared(create_vec(self.n_hidden))
        self.params_p = [self.W, self.vbias, self.hbias]

        # running averages
        self.inner_it = theano.shared(create_sca(0.0))
        self.loga = theano.shared(create_sca(0.0))
        self.logZ_avg = theano.shared(create_sca(0.0))
        self.grad_logZ_avg = [
          theano.shared(create_zmat(self.n_visible, self.n_hidden)), 
          theano.shared(create_vec(self.n_visible)),
          theano.shared(create_vec(self.n_hidden))
        ]
        self.grad_p_avg = [
          theano.shared(create_zmat(self.n_visible, self.n_hidden)), 
          theano.shared(create_vec(self.n_visible)),
          theano.shared(create_vec(self.n_hidden))
        ]
        self.W_avg = theano.shared(self.W.get_value().copy())
        self.vbias_avg = theano.shared(self.vbias.get_value().copy())
        self.hbias_avg = theano.shared(self.hbias.get_value().copy())

        # for pseudo-likelihood
        self.bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # create approximating distribution
        n_aux = 10 # dimensionality of latent auxiliary space
        self.A = self.theano_rng.normal(size=(self.n_mc, n_aux))

        # create q(x|a)
        l_qx_in = lasagne.layers.InputLayer((None, n_aux))

        # digits dataset
        if self.n_visible == 64:
            # l_qx_hid = lasagne.layers.DenseLayer(l_qx_in, num_units=64, 
            #   nonlinearity=lasagne.nonlinearities.rectify)
            l_qx_hid = lasagne.layers.DenseLayer(l_qx_in, num_units=32, 
              nonlinearity=lasagne.nonlinearities.rectify)
            l_qx = lasagne.layers.DenseLayer(l_qx_hid, num_units=self.n_latent, 
              nonlinearity=lasagne.nonlinearities.sigmoid)
            l_qx_samp = BernoulliSampleLayer(l_qx)

            # l_g_hid1 = lasagne.layers.DenseLayer(l_qx_in, 8*4*4)
            # l_g_hid2 = lasagne.layers.ReshapeLayer(l_g_hid1, ([0], 8, 4, 4))
            # # l_g_dc1 = Deconv2DLayer(l_g_hid2, 8, 3, stride=2, pad=1)
            # l_g = Deconv2DLayer(l_g_hid2, 1, 3, stride=2, pad=1, 
            #         nonlinearity=lasagne.nonlinearities.sigmoid)
            # l_qx = lasagne.layers.flatten(l_g)
            # l_qx_samp = BernoulliSampleLayer(l_qx)

        # mnist dataset
        elif self.n_visible == 784:
            # dense
            l_qx_hid = lasagne.layers.DenseLayer(l_qx_in, num_units=2048, 
              nonlinearity=lasagne.nonlinearities.rectify)
            # l_qx_hid = batch_norm(lasagne.layers.DenseLayer(l_qx_in, num_units=128, 
            #   nonlinearity=lasagne.nonlinearities.rectify))
            # l_qx_hid = batch_norm(lasagne.layers.DenseLayer(l_qx_hid, num_units=256, 
            #   nonlinearity=lasagne.nonlinearities.rectify))
            l_qx = lasagne.layers.DenseLayer(l_qx_hid, num_units=self.n_latent, 
              nonlinearity=lasagne.nonlinearities.sigmoid)
            l_qx_samp = BernoulliSampleLayer(l_qx)

            # # deconv
            # l_g_hid1 = batch_norm(lasagne.layers.DenseLayer(l_qx_in, 64*7*7))
            # l_g_hid2 = lasagne.layers.ReshapeLayer(l_g_hid1, ([0], 64, 7, 7))
            # # l_g_dc = Deconv2DLayer(l_g_hid2, 1, 5, stride=4, pad=2, 
            # #         nonlinearity=lasagne.nonlinearities.sigmoid)

            # # l_g_dc1 = Deconv2DLayer(l_g_hid2, 8, 3, stride=2, pad=1)
            # l_g_dc1 = batch_norm(Deconv2DLayer(l_g_hid2, 32, 5, stride=2, pad=2, 
            #         nonlinearity=lasagne.nonlinearities.rectify))
            # l_g_dc = Deconv2DLayer(l_g_dc1, 1, 5, stride=2, pad=2, 
            #         nonlinearity=lasagne.nonlinearities.sigmoid)
            
            # l_qx = lasagne.layers.flatten(l_g_dc)
            # l_qx_samp = BernoulliSampleLayer(l_qx)

        # create p(a|x)
        relu_shift = lambda av: T.nnet.relu(av+10)-10 # for numerical stability
        l_pa_in = lasagne.layers.InputLayer((None, self.n_latent))
        l_pa_hid = lasagne.layers.DenseLayer(l_pa_in, num_units=32, 
          nonlinearity=lasagne.nonlinearities.rectify)
        l_pa_mu = lasagne.layers.DenseLayer(l_pa_hid, num_units=n_aux, 
          nonlinearity=None)
        l_pa_logsigma = lasagne.layers.DenseLayer(l_pa_hid, num_units=n_aux, 
          nonlinearity=relu_shift)

        return (l_qx, l_qx_samp, l_pa_mu, l_pa_logsigma)

    def _create_components(self, D):
        # collect samples
        (l_qx, l_qx_samp, l_pa_mu, l_pa_logsigma) = self.network
        a = self.A
        qx, x = lasagne.layers.get_output([l_qx, l_qx_samp], a)
        pa_mu, pa_logsigma = lasagne.layers.get_output([l_pa_mu, l_pa_logsigma], x)

        # compute logQ
        logQa = T.sum(log_normal(a, 0., 1.), axis=1)
        logQx_given_a = T.sum(log_bernoulli(x, qx), axis=1)
        logQ = logQa + logQx_given_a

        # compute energies of the samples, dim=(1, n_tot_samples)
        logFx = self._free_energy(x.T, marginalize=self.marginalize)
        logpa = T.sum(log_normal2(a, pa_mu, pa_logsigma), axis=1)
        # logF = logFx + logpa

        # free energy of the data
        D = D.reshape((-1, self.n_visible)).T
        logF_D = self._free_energy(D)

        self._components = (logFx, logpa, logQ, logF_D)

    def create_objectives(self, D, deterministic=False):
        self._create_components(D)
        (logFx, logpa, logQ, logF_D_vec) = self._components
        t = self.n_mc
        logFxa = logFx + logpa

        # set up bound
        logFQ = logFx - logQ # (t,)
        logFQ2 = 2.*logFxa - 2.*logQ # (t,)
        logZ = Tlogsumexp(logFQ.flatten(), axis=0) - T.log(t)
        logbnd = Tlogsumexp(logFQ2.flatten(), axis=0) - T.log(t)

        logF_D = logF_D_vec.mean()

        loglik = logF_D - 0.5*logZ
        loglikbnd = logF_D - 0.5*logbnd

        self.logbnd = logbnd
        self.logZ = logZ

        # set up pseudolikelihood
        plik = self.create_pseudoliklihood(D)

        return loglik, plik

    # def get_params(self):
    #     return [self.W, self.vbias, self.hbias, self.Phi]

    def get_params_p(self):
        return [self.W, self.vbias, self.hbias]

    def get_params_qx(self):
        (l_qx, l_qx_samp, l_pa_mu, l_pa_logsigma) = self.network
        return lasagne.layers.get_all_params(l_qx, trainable=True)

    def get_params_pa(self):
        (l_qx, l_qx_samp, l_pa_mu, l_pa_logsigma) = self.network
        return lasagne.layers.get_all_params(l_pa_mu, trainable=True)

    def create_gradients(self):
        (logFx, logpa, logQ, logF_D_vec) = self._components
        logFxa = logFx + logpa
        logbnd, logZ = self.logbnd, self.logZ
        logF_D = logF_D_vec.mean()
        
        logQ2 = 2.*logQ
        logFQ = logFx - logQ # (t,)
        S = T.exp(logFQ - logZ) # (t,)
        S2 = T.exp(2*logFxa - logbnd - logQ2) # (t,)
        target = T.mean(S * logFx)

        # get grads wrt params_p
        dlogB_W, dlogB_bv, dlogB_bh = theano.grad(logbnd, [self.W, self.vbias, self.hbias])
        dlogZ_W, dlogZ_bv, dlogZ_bh = theano.grad(target, [self.W, self.vbias, self.hbias], consider_constant=[S])
        dE_W, dE_bv, dE_bh = theano.grad(logF_D, [self.W, self.vbias, self.hbias])

        # get grads wrt params_qx
        from theano.gradient import disconnected_grad as dg
        dlogB_target = T.mean(-dg(S2) * logQ)
        dlogB_qx = theano.grad(dlogB_target, self.get_params_qx())

        # get grads wrt params_pa
        dlogB_pa = theano.grad(T.mean(S2), self.get_params_pa(), consider_constant=[logbnd])

        # log-likelihood / bound gradients (combine the above)
        
        dL_qx = [-0.5*g for g in dlogB_qx]
        dL_pa = [-0.5*g for g in dlogB_pa]
        dL_W = dE_W - dlogZ_W 
        dL_bv = dE_bv - dlogZ_bv
        dL_bh = dE_bh - dlogZ_bh
        # dL_W = dE_W - 0.5 * dlogB_W 
        # dL_bv = dE_bv - 0.5 * dlogB_bv
        # dL_bh = dE_bh - 0.5 * dlogB_bh

        dL_Theta = [dL_W, dL_bv, dL_bh]
        dlogZ_Theta = [dlogZ_W, dlogZ_bv, dlogZ_bh]
        dE_Theta = [dE_W, dE_bv, dE_bh]

        return dL_Theta, dlogZ_Theta, dE_Theta, dL_qx, dL_pa

    def create_pseudoliklihood(self, X):
        """Stochastic approximation to the pseudo-likelihood"""
        # X = self.inputs[0]
        X = X.reshape((-1, self.n_visible))

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = self.bit_i_idx
        # bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(X)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip-fe_xi)))

        return cost

    def create_updates(self, grads, params, alpha, opt_alg, opt_params):
    # def create_updates(self, alpha, opt_alg, opt_params):
        scaled_p_grads = [grad * alpha for grad in self.grad_p_avg]
        dL_Theta, dE_Theta, dlogZ_Theta, dL_qx, dL_pa = grads

        # create update for main parameters
        # TODO: Scale grads with alpha!
        # lr = opt_params.get('lr', 1e-3)
        # b1, b2 = opt_params.get('b1', 0.7), opt_params.get('b2', 1e-3)
        lr, b1, b2 = 1e-3, 0.7, 1e-3
        uL_Theta = lasagne.updates.adam(self.grad_p_avg, self.params_p, learning_rate=lr, beta1=b1, beta2=b2)
        # uL_Theta = lasagne.updates.adam(self.grad_p_avg, self.params_p, learning_rate=1e-3, beta1=7e-1, beta2=1e-3)
        # uL_Theta = lasagne.updates.adam(scaled_p_grads, self.params_p, learning_rate=1e-3, beta1=0.9, beta2=0.999)
          # learning_rate=lr, beta1=b1, beta2=b2)
        uL_qx = lasagne.updates.sgd([-1*g for g in dL_qx], self.get_params_qx(), learning_rate=5e-2) # TODO: anneal lr
        # uL_qx = lasagne.updates.adam([-1*g for g in dL_qx], self.get_params_qx(), learning_rate=5e-2, beta1=0.0, beta2=0.999) # TODO: anneal lr
        uL_pa = lasagne.updates.sgd([-1*g for g in dL_pa], self.get_params_pa(), learning_rate=5e-2) # TODO: anneal lr
        # uL_Phi = OrderedDict(uL_qx.items() + uL_pa.items())
        uL_Phi = uL_qx

        # create updates for the running averages
        avg_updates = self.create_avg_updates(self.logZ, dlogZ_Theta, dL_Theta)

        # avg Theta updates
        avg_Theta_updates = OrderedDict()
        avg_Theta_updates[self.W_avg] = 0.95*self.W_avg + 0.05*self.W
        avg_Theta_updates[self.vbias_avg] = 0.95*self.vbias_avg + 0.05*self.vbias
        avg_Theta_updates[self.hbias_avg] = 0.95*self.hbias_avg + 0.05*self.hbias
        # avg_Theta_updates[self.W_avg] = self.W
        # avg_Theta_updates[self.vbias_avg] = self.vbias
        # avg_Theta_updates[self.hbias_avg] = self.hbias

        # increment bit_i_idx % number as part of updates
        uL_Phi[self.bit_i_idx] = (self.bit_i_idx + 1) % self.n_visible

        return uL_Theta, uL_Phi, avg_updates, avg_Theta_updates

    def create_avg_updates(self, logZ, dlogZ_Theta, dL_Theta):
      inner_it = self.inner_it
      avg_updates = OrderedDict()

      new_logZ_estimate = Tlogaddexp(
          self.logZ_avg + T.log((inner_it) / (inner_it+1)),
          logZ - T.log(inner_it+1.)
      )
        # self.logZ_avg * ((inner_it) / (inner_it+1)) + logZ / (inner_it+1)

      avg_updates[inner_it] = inner_it + 1.
      avg_updates[self.logZ_avg] = new_logZ_estimate
      avg_updates[self.loga] = -2.*new_logZ_estimate # 1./(new_Z_estimate**2)
      
      for g, g_avg in zip(dL_Theta, self.grad_p_avg):
        avg_updates[g_avg] = g_avg * ((inner_it) / (inner_it+1)) - g / (inner_it+1)

      return avg_updates

    def reset_averages(self):
      self.inner_it.set_value(0.0)

    def free_energy(self, v_sample):
        """Function to compute the free energy"""
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def pseudolikelihood(self, data):
        self.rbm.components_ = self.W.get_value().T
        self.rbm.intercept_visible_ = self.vbias.get_value()
        self.rbm.intercept_hidden_  = self.hbias.get_value()
        return self.rbm.score_samples(data).mean()

    def dump(self, fname):
        """Pickle weights to a file"""
        params = lasagne.layers.get_all_param_values(self.network)
        all_params = [params, self.W.get_value(), self.vbias.get_value(), self.hbias.get_value()]
        with open(fname, 'w') as f:
            pickle.dump(all_params, f)

    def load(self, fname):
        """Load pickled network"""
        with open(fname) as f:
            params = pickle.load(f)
        self.load_params(params)

    def load_params(self, params):
        """Load a given set of parameters"""
        net_params, W, vbias, hbias = params
        self.W.set_value(W)
        self.vbias.set_value(vbias)
        self.hbias.set_value(hbias)
        lasagne.layers.set_all_param_values(self.network, net_params)

    def fit(
        self, X_train, Y_train, X_val, Y_val,
        n_epoch=10, n_batch=100, logname='run'
    ):
        """Train the model"""
        alpha = 1.0 # learning rate, which can be adjusted later
        n_data = len(X_train)
        n_superbatch = self.n_superbatch

        for epoch in range(n_epoch):
            # In each epoch, we do a full pass over the training data:
            train_batches, train_err, train_acc = 0, 0, 0
            start_time = time.time()

            # if epoch > 30: alpha = 0.33
            # if epoch > 20: alpha = 0.1

            # iterate over superbatches to save time on GPU memory transfer
            for X_sb, Y_sb in self.iterate_superbatches(
                X_train, Y_train, n_superbatch,
                datatype='train', shuffle=True,
            ):
                for idx1, idx2 in iterate_minibatch_idx(len(X_sb), n_batch):
                    # q steps
                    for i in range(2):
                      err, acc = self.train_q(idx1, idx2)
                      # print epoch, i, err, acc
                    # err, acc = self.train_q(idx1, idx2)

                    # p steps
                    err, acc = self.train_p(idx1, idx2, alpha)
                    # print epoch, 'P', self.pseudolikelihood(X_sb[idx1:idx2].reshape(-1, self.n_visible)), acc
                    # print
                    self.reset_averages()

                    # err, acc = self.train(idx1, idx2, alpha)
                    # collect metrics
                    err = self.pseudolikelihood(X_sb[idx1:idx2].reshape(-1, self.n_visible))
                    train_batches += 1
                    train_err += err
                    train_acc += acc

                    # print train_batches, err, acc

                    if train_batches % 100 == 0:
                        n_total = epoch * n_data + n_batch * train_batches
                        metrics = [
                            n_total, train_err / train_batches,
                            train_acc / train_batches,
                        ]
                        log_metrics(logname, metrics)

            print "Epoch {} of {} took {:.3f}s ({} minibatches)".format(
                epoch + 1, n_epoch, time.time() - start_time, train_batches)

            print "  training:\t\t{:.6f}\t{:.6f}".format(
                train_err / train_batches, train_acc / train_batches)

        # reserve N of training data points to kick start hallucinations
        self.dump(logname + '.pkl')
        hallu_i = self.numpy_rng.randint(n_data - self.n_chain)
        self.hallu_set = np.asarray(
            X_train[hallu_i:hallu_i + self.n_chain],
            dtype=theano.config.floatX
        )