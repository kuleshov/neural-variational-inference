import pdb
import time
import pickle
import numpy as np
import theano
import lasagne
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict

from model import Model
from abstractrbm import AbstractRBM
from helpers import iterate_minibatch_idx, evaluate, log_metrics
from helpers import Tlogsumexp, Tlogaddexp

from theano.compile.nanguardmode import NanGuardMode

create_zmat = lambda x, y: np.zeros((x, y)).astype(theano.config.floatX)
create_vec = lambda y: np.zeros(y,).astype(theano.config.floatX)
create_sca = lambda x: (np.cast[theano.config.floatX](x)).astype(theano.config.floatX)

class VariationalRBM(AbstractRBM):
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
        self.n_hidden = 100
        self.n_visible = n_chan*n_dim*n_dim  # size of visible layer
        self.n_batch = n_batch
        self.n_qk = 10 # num of components in MoB used of q
        self.n_mc = 30 # num of monte carlo samples from each MoB component

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


        self.n_components = self.n_qk
        self.n_samples = self.n_mc
        self.n_tot_samples = self.n_samples*self.n_components


        # create input variables
        D, idx1, idx2 = self.create_inputs()

        # create model
        self.network = self.create_model()

        # create objectives
        loglik, plik = self.create_objectives(D)

        # create gradients
        dL_Theta, dE_Theta, dlogZ_Theta, dL_Phi = self.create_gradients()
        grads = dL_Theta, dE_Theta, dlogZ_Theta, dL_Phi

        # create updates
        uL_Theta, uL_Phi, avg_updates, avg_Theta_updates \
          = self.create_updates(grads, None, alpha, opt_alg, opt_params)
      
        # logF_avg, Z_avg = self.create_llik_estimate(D)
        
        mode = NanGuardMode(nan_is_error=True, inf_is_error=False, big_is_error=False)
        mode = None

        common_update1 = OrderedDict(avg_updates.items() + uL_Phi.items())
        self.train_q = theano.function([idx1, idx2], [loglik, plik], 
          updates=common_update1, mode=mode,
          givens={D: self.train_set_x[idx1:idx2]})

        common_update2 = OrderedDict(avg_Theta_updates.items() + uL_Theta.items())
        self.train_p = theano.function([idx1, idx2], [loglik, plik], 
            updates=common_update2, mode=mode, on_unused_input='warn',
            givens={D: self.train_set_x[idx1:idx2]})
        # self.llik = theano.function([D], logF_avg - T.log(Z_avg), mode=mode)

        common_update3 = OrderedDict(common_update1.items() + common_update2.items())
        self.train = theano.function([idx1, idx2], [loglik, plik], 
            updates=common_update3, mode=mode,
            givens={D: self.train_set_x[idx1:idx2]})

    def create_inputs(self):
        # allocate symbolic variables for the data
        idx1, idx2 = T.lscalar(), T.lscalar()
        D = T.tensor4('D', dtype=theano.config.floatX) # data matrixX)  # learning rate
        # self.Z = T.tensor2(dtype=theano.config.floatX)
        return D, idx1, idx2

    def _free_energy(self, X, avg=False):
      bv_vec = self.vbias.reshape((self.n_visible,1))
      bh_vec = self.hbias.reshape((self.n_hidden,1))

      W = self.W_avg if avg else self.W
      W = W.T
      
      logF = T.dot(bv_vec.T, X) \
             + T.sum(T.log(1. + T.exp(bh_vec + T.dot(W, X))), axis=0)
      
      return logF

    @property
    def Z(self):
      return np.exp(self.logZ_avg.get_value())

    def create_model(self):
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

        self.bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # create approximating distribution
        create_mat = lambda x, y: self.numpy_rng.normal(0, 0.01, (x, y)).astype(theano.config.floatX)
        self.pi = np.full( (self.n_qk,1), 1./float(self.n_qk) ).astype(theano.config.floatX)
        self.Phi = theano.shared(create_mat(self.n_visible,self.n_qk))
        self.params_q = [self.Phi]

        # create symbolic variables representing the samples
        Qk = 1./(1+T.exp(-self.Phi)) # (n,k) component probabilities
        Qt = T.tile(T.reshape(Qk, (self.n_visible, self.n_qk, 1)), (1, 1, self.n_mc))
        Qs = self.theano_rng.binomial(size=Qt.shape, p=Qt, dtype=Qt.dtype)
        self.q_samples = Qs
        self.Qk = Qk

        return None

    def _create_components(self, D):
        # shorthands
        k, t, s = self.n_components, self.n_tot_samples, self.n_samples
        n = self.n_visible

        ## compute bound objective

        X = self.q_samples
        D = D.reshape((-1, self.n_visible)).T
        X2 = T.flatten(X, outdim=2)

        # compute the probability of each datapoint under each component

        # reshape tensors
        QQ = self.Qk.reshape((n,k,1))
        XX = X2.dimshuffle(0,'x',1)
        self.XX = XX
        T.addbroadcast(QQ, 2)
        T.addbroadcast(XX, 1)

        # compute the probabilities (log version is 2x slower...)
        # logQ_prob = T.log(QQ*XX + (1.-QQ)*(1.-XX)) # (n, k, t)
        # logQ_tot = T.sum(logQ_prob, axis=0) # (k, t)
        # logQ = Tlogsumexp(np.log(self.pi_uniform) + logQ_tot, axis=0) # (t,)
        Q_tot = T.prod(QQ*XX + (1-QQ)*(1-XX), axis=0)
        Q = T.sum(self.pi * Q_tot, axis=0) # (t,)
        logQ = T.log(Q)

        # compute energies of the samples, dim=(1, n_tot_samples)
        logF = self._free_energy(X2)

        # free energy of the data
        logF_D = self._free_energy(D)

        self._components = (logF, logQ, logF_D)

    def create_objectives(self, D, deterministic=False):
        self._create_components(D)
        (logF, logQ, logF_D_vec) = self._components
        t = self.n_qk * self.n_mc

        # set up bound
        t = self.n_qk * self.n_mc
        logFQ = logF - logQ # (t,)
        logFQ2 = 2.*logF - 2.*logQ # (t,)
        logFQ = logFQ.flatten()
        logFQ2 = logFQ2.flatten()
        logZ = Tlogsumexp(logFQ, axis=0) - T.log(t)
        logbnd = Tlogsumexp(logFQ2, axis=0) - T.log(t)

        logF_D = logF_D_vec.mean()

        loglik = logF_D - 0.5*logZ
        loglikbnd = logF_D - 0.5*logbnd

        self.logbnd = logbnd
        self.logZ = logZ

        # set up pseudolikelihood
        plik = self.create_pseudoliklihood(D)

        return loglik, plik

    def get_params(self):
        return [self.W, self.vbias, self.hbias, self.Phi]

    def get_params_p(self):
        return [self.W, self.vbias, self.hbias]

    def get_params_q(self):
        return [self.Phi]

    def create_gradients(self):
        k, s, n = self.n_qk, self.n_mc, self.n_visible
        (logF, logQ, logF_D_vec) = self._components
        logbnd, logZ = self.logbnd, self.logZ
        logF_D = logF_D_vec.mean()
        X = self.q_samples
        QQ = self.Qk.reshape((n,k,1))
        
        logF2 = 2.*logF
        logQ2 = 2.*logQ
        logFQ = logF - logQ # (t,)
        S = T.exp(logFQ - logZ) # (t,)
        S2 = T.exp(logF2 - logbnd - logQ2) # (t,)
        target = T.mean(S * logF)

        # get grads wrt params_p
        dlogB_W, dlogB_bv, dlogB_bh = theano.grad(logbnd, [self.W, self.vbias, self.hbias])
        dlogZ_W, dlogZ_bv, dlogZ_bh = theano.grad(target, [self.W, self.vbias, self.hbias], consider_constant=[S])
        dE_W, dE_bv, dE_bh = theano.grad(logF_D, [self.W, self.vbias, self.hbias])

        # get graps wrt params_q
        loga = - logbnd
        cv = T.exp(logZ + 0.5*loga)
        logF2a = logF2 + loga
        F2a = T.exp( logF2a )
        Q2 = T.exp(logQ2)
        cv_adj = cv**2. * Q2
        Scv = (F2a - cv_adj)/Q2
        # S = (F2a)/Q2
        # S = FQ2a

        Dq = X - QQ # (n, K, s)
        Dq *= (-Scv).reshape((1,k,s))
        dlogB_Phi = T.mean(Dq, axis=2) * self.pi.reshape(1,k) # (n, k)

        from theano.gradient import disconnected_grad as dg
        dlogB_target = T.mean(-dg(S2) * logQ)
        dlogB_Phi = theano.grad(dlogB_target, self.Phi, consider_constant=[self.XX])        

        # log-likelihood / bound gradients (combine the above)
        
        # dL_Phi = -0.5*a*dB_Phi
        dL_Phi = -0.5*dlogB_Phi
        dL_W = dE_W - dlogZ_W 
        dL_bv = dE_bv - dlogZ_bv
        dL_bh = dE_bh - dlogZ_bh
        # dL_W = dE_W - 0.5 * dlogB_W 
        # dL_bv = dE_bv - 0.5 * dlogB_bv
        # dL_bh = dE_bh - 0.5 * dlogB_bh

        dL_Theta = [dL_W, dL_bv, dL_bh]
        dlogZ_Theta = [dlogZ_W, dlogZ_bv, dlogZ_bh]
        dE_Theta = [dE_W, dE_bv, dE_bh]

        return dL_Theta, dlogZ_Theta, dE_Theta, dL_Phi

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
        # scaled_grads = [grad * alpha for grad in grads]
        # scaled_grads = [T.clip(grad, -1., 1.) for grad in scaled_grads]
        dL_Theta, dE_Theta, dlogZ_Theta, dL_Phi = grads

        # create update for main parameters
        # TODO: Scale grads with alpha!
        lr = opt_params.get('lr', 1e-3)
        b1, b2 = opt_params.get('b1', 0.9), opt_params.get('b2', 0.999)
        uL_Theta = lasagne.updates.adam(self.grad_p_avg, self.params_p, learning_rate=1e-3, beta1=7e-1, beta2=1e-3)
          # learning_rate=lr, beta1=b1, beta2=b2)
        uL_Phi = lasagne.updates.sgd([-dL_Phi], [self.Phi], learning_rate=5e-1) # TODO: anneal lr

        # create updates for the running averages
        avg_updates = self.create_avg_updates(self.logZ, dlogZ_Theta, dL_Theta)

        # avg Theta updates
        avg_Theta_updates = OrderedDict()
        avg_Theta_updates[self.W_avg] = 0.95*self.W_avg + 0.05*self.W
        avg_Theta_updates[self.vbias_avg] = 0.95*self.vbias_avg + 0.05*self.vbias
        avg_Theta_updates[self.hbias_avg] = 0.95*self.hbias_avg + 0.05*self.hbias

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
        all_params = [self.W.get_value(), self.vbias.get_value(), self.hbias.get_value(), self.Phi.get_value()]
        with open(fname, 'w') as f:
            pickle.dump(all_params, f)

    def load(self, fname):
        """Load pickled network"""
        with open(fname) as f:
            params = pickle.load(f)
        self.load_params(params)

    def load_params(self, params):
        """Load a given set of parameters"""
        W, vbias, hbias, Phi = params
        self.W.set_value(W)
        self.vbias.set_value(vbias)
        self.hbias.set_value(hbias)
        self.Phi.set_value(Phi)

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

            # if epoch > 100: alpha = 0.5

            # iterate over superbatches to save time on GPU memory transfer
            for X_sb, Y_sb in self.iterate_superbatches(
                X_train, Y_train, n_superbatch,
                datatype='train', shuffle=True,
            ):
                for idx1, idx2 in iterate_minibatch_idx(len(X_sb), n_batch):
                    # q steps
                    err, acc = self.train_q(idx1, idx2)
                    err, acc = self.train_q(idx1, idx2)
                    err, acc = self.train_q(idx1, idx2)

                    # p steps
                    err, acc = self.train_p(idx1, idx2)
                    self.reset_averages()

                    # err, acc = self.train(idx1, idx2)
                    # self.reset_averages()
                    # collect metrics
                    err = self.pseudolikelihood(X_sb[idx1:idx2].reshape(-1, 64))
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