import pdb
import time
import numpy as np
import theano
import lasagne
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from model import Model
from helpers import iterate_minibatch_idx, evaluate, log_metrics

from sklearn.neural_network import BernoulliRBM


class AbstractRBM(Model):
    """Restricted Boltzmann Machine
    RBM code adapted from http://deeplearning.net/tutorial/rbm.html

    Epoch 15 of 15 took 635.792s (2448 minibatches)
        training loss/acc:    -62.311016  -62.311016

    Training Params
    ---------------
    batch_size: 20
    learning_rate: 0.1
    """
    def __init__(
        self, n_dim, n_out, n_chan=1, n_superbatch=12800, opt_alg='adam',
        opt_params={'lr': 1e-3, 'b1': 0.9, 'b2': 0.99}
    ):
        """RBM constructor.
        Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.
        """

        # store sklearn RBM instance
        self.rbm = BernoulliRBM(random_state=0, n_components=self.n_hidden)

        self.n_chain = 100

        # initialize storage for the persistent chain (state = hidden
        # layer of chain)
        self.persistent_chain = theano.shared(
            np.zeros(
                (self.n_batch, self.n_hidden),
                dtype=theano.config.floatX
            ), borrow=True,
        )

        self.bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # create shared data variables
        self.train_set_x = theano.shared(np.empty(
            (n_superbatch, n_chan, n_dim, n_dim),
            dtype=theano.config.floatX),
            borrow=False,
        )
        self.val_set_x = theano.shared(np.empty(
            (n_superbatch, n_chan, n_dim, n_dim),
            dtype=theano.config.floatX),
            borrow=False,
        )

        # create y-variables
        self.train_set_y = theano.shared(np.empty(
            (n_superbatch,), dtype='int32'), borrow=False)
        self.val_set_y = theano.shared(np.empty(
            (n_superbatch,), dtype='int32'), borrow=False)
        # train_set_y_int = T.cast(train_set_y, 'int32')
        # val_set_y_int = T.cast(val_set_y, 'int32')

    def free_energy(self, v_sample):
        """Function to compute the free energy"""
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup(self, vis):
        """This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)
        """
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        """This function infers state of hidden units given visible units"""
        # compute the activation of the hidden units given a sample of
        # the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(
            size=h1_mean.shape,
            n=1, p=h1_mean,
            dtype=theano.config.floatX,
        )
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        """This function propagates the hidden units activation downwards to
        the visible units

        Note that we return also the pre_sigmoid_activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)
        """
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        """This function infers state of visible units given hidden units"""
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(
            size=v1_mean.shape,
            n=1, p=v1_mean,
            dtype=theano.config.floatX,
        )
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        """This function implements one step of Gibbs sampling,
        starting from the hidden state
        """
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        """This function implements one step of Gibbs sampling,
        starting from the visible state
        """
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def pseudolikelihood(self, data):
        self.rbm.components_ = self.W.get_value().T
        self.rbm.intercept_visible_ = self.vbias.get_value()
        self.rbm.intercept_hidden_  = self.hbias.get_value()
        return self.rbm.score_samples(data).mean()

    def get_pseudo_likelihood_cost(self, X, updates):
        """Stochastic approximation to the pseudo-likelihood"""
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

    def hallucinate(self):
        """Once the RBM is trained, we can then use the gibbs_vhv function to
        implement the Gibbs chain required for sampling. This overwrites the
        hallucinate function in Model completely.
        """
        n_samples = 10
        hallu_set = self.hallu_set.reshape((-1, self.n_visible))
        persistent_vis_chain = theano.shared(hallu_set)
        # define one step of Gibbs sampling (mf = mean-field) define a
        # function that does `1000` steps before returning the
        # sample for plotting
        (
            [
                presig_hids,
                hid_mfs,
                hid_samples,
                presig_vis,
                vis_mfs,
                vis_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_vhv,
            outputs_info=[None, None, None, None, None, persistent_vis_chain],
            n_steps=1000,
            name="gibbs_vhv",
        )

        # add to updates that takes care of our persistent chain :
        updates.update({persistent_vis_chain: vis_samples[-1]})
        # construct the function that implements our persistent chain.
        # we generate the "mean field" activations for plotting and the actual
        # samples for reinitializing the state of our persistent chain
        sample_fn = theano.function(
            [],
            [
                vis_mfs[-1],
                vis_samples[-1]
            ],
            updates=updates,
            name='sample_fn',
        )

        for idx in range(n_samples):
            # generate `plot_every` intermediate samples that we discard,
            # because successive samples in the chain are too correlated
            vis_mf, vis_sample = sample_fn()

        img_size = int(np.sqrt(self.n_chain))
        vis_mf = vis_mf.reshape((img_size, img_size, self.n_dim, self.n_dim))
        vis_mf = np.concatenate(np.split(vis_mf, img_size, axis=0), axis=3)
        # split into img_size (1,1,n_dim,n_dim*img_size) images,
        # concat along rows -> 1,1,n_dim*img_size,n_dim*img_size
        vis_mf = np.concatenate(np.split(vis_mf, img_size, axis=1), axis=2)
        return np.squeeze(vis_mf)

    def hallucinate_chain(self):
        """Once the RBM is trained, we can then use the gibbs_vhv function to
        implement the Gibbs chain required for sampling. This overwrites the
        hallucinate function in Model completely.
        """
        n_samples = 10
        # hallu_set = self.hallu_set.reshape((-1, self.n_visible))
        hallu_set = np.random.rand(1, 64).astype('float32')
        # hallu_set = self.Phi.get_value()[:,0]
        persistent_vis_chain = theano.shared(hallu_set)
        # define one step of Gibbs sampling (mf = mean-field) define a
        # function that does `1000` steps before returning the
        # sample for plotting
        (
            [
                presig_hids,
                hid_mfs,
                hid_samples,
                presig_vis,
                vis_mfs,
                vis_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_vhv,
            outputs_info=[None, None, None, None, None, persistent_vis_chain],
            n_steps=100,
            name="gibbs_vhv",
        )

        # add to updates that takes care of our persistent chain :
        updates.update({persistent_vis_chain: vis_samples[-1]})
        # construct the function that implements our persistent chain.
        # we generate the "mean field" activations for plotting and the actual
        # samples for reinitializing the state of our persistent chain
        sample_fn = theano.function([],vis_mfs,updates=updates,name='sample_fn')

        vis_mf = sample_fn()
        print vis_mf.shape
        # vis_mf = vis_mf[::10]
        print vis_mf.shape

        img_size = 10
        vis_mf = vis_mf.reshape((img_size, img_size, self.n_dim, self.n_dim))
        vis_mf = np.transpose(vis_mf, [1,0,2,3])
        vis_mf = np.concatenate(np.split(vis_mf, img_size, axis=0), axis=3)
        # split into img_size (1,1,n_dim,n_dim*img_size) images,
        # concat along rows -> 1,1,n_dim*img_size,n_dim*img_size
        vis_mf = np.concatenate(np.split(vis_mf, img_size, axis=1), axis=2)
        return np.squeeze(vis_mf)

    def E_np_h(self,h):
        bv_vec = self.vbias.get_value().reshape((self.n_visible,1))
        bh_vec = self.hbias.get_value().reshape((self.n_hidden,1))
        W = self.W.get_value()
        return (np.dot(bh_vec.T, h) + np.sum(np.log(1. + np.exp(bv_vec + np.dot(W, h))), axis=0)).flatten()

    def E_np_v(self,v):
        bv_vec = self.vbias.get_value().reshape((self.n_visible,1))
        bh_vec = self.hbias.get_value().reshape((self.n_hidden,1))
        W = self.W.get_value()
        return (np.dot(bv_vec.T, v)+ np.sum(np.log(1. + np.exp(bh_vec + np.dot(W.T, v))), axis=0)).flatten()

    def logZ_exact(self, marg='v'):
        # get the next binary vector
        t = self.n_hidden if marg == 'v' else self.n_visible
        def inc(x):
            for i in xrange(t):
                x[i,0]+=1
                if x[i,0]<=1: return True
                x[i,0]=0

            return False

        #compute the normalizing constant
        if marg == 'v': x=np.zeros((self.n_hidden,1))
        elif marg == 'h': x=np.zeros((self.n_visible,1))
        logZ=-np.inf
        while True:
            if marg == 'v': logF = self.E_np_h(x)
            elif marg == 'h': logF = self.E_np_v(x)
            # print ''.join([str(xi) for xi in x]), logF, logZ
            logZ=np.logaddexp(logZ,logF)
            if not inc(x): break

        # print
        return logZ

    def sample(self):
        """Once the RBM is trained, we can then use the gibbs_vhv function to
        implement the Gibbs chain required for sampling. This overwrites the
        hallucinate function in Model completely.
        """
        n_samples = 10
        hallu_set = self.hallu_set.reshape((-1, self.n_visible))
        persistent_vis_chain = theano.shared(hallu_set)
        # define one step of Gibbs sampling (mf = mean-field) define a
        # function that does `1000` steps before returning the
        # sample for plotting
        (
            [
                presig_hids,
                hid_mfs,
                hid_samples,
                presig_vis,
                vis_mfs,
                vis_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_vhv,
            outputs_info=[None, None, None, None, None, persistent_vis_chain],
            n_steps=1000,
            name="gibbs_vhv",
        )

        # add to updates that takes care of our persistent chain :
        updates.update({persistent_vis_chain: vis_samples[-1]})
        # construct the function that implements our persistent chain.
        # we generate the "mean field" activations for plotting and the actual
        # samples for reinitializing the state of our persistent chain
        sample_fn = theano.function(
            [],
            [
                vis_mfs[-1],
                vis_samples[-1]
            ],
            updates=updates,
            name='sample_fn',
        )

        for idx in range(n_samples):
            # generate `plot_every` intermediate samples that we discard,
            # because successive samples in the chain are too correlated
            vis_mf, vis_sample = sample_fn()

        img_size = int(np.sqrt(self.n_chain))
        vis_mf = vis_mf.reshape((self.n_chain, self.n_dim*self.n_dim))

        return vis_mf