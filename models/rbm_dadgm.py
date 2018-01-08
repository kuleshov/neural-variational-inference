import numpy as np
import theano.tensor as T
from lasagne.layers import (
  InputLayer, DenseLayer, ElemwiseSumLayer,
  reshape, flatten, get_all_params, get_output,
)
from lasagne.updates import total_norm_constraint
from lasagne.init import GlorotNormal, Normal
from layers import GumbelSoftmaxSampleLayer, GaussianSampleLayer
from distributions import log_bernoulli, log_normal2
from gsm import GSM
from rbm import RBM

import theano
import lasagne
from theano.tensor.shared_randomstreams import RandomStreams


class RBM_DADGM(GSM, RBM):
    """Restricted Boltzmann Machine trained with persistent contrastive
    divergence for learning p; Auxiliary Deep Generative Model trained
    using Gumbel Softmax Reparametrization for learning q.
    """

    def __init__(
        self, n_dim, n_out, n_chan=1, n_superbatch=12800, opt_alg='adam',
        opt_params={'lr': 1e-3, 'b1': 0.9, 'b2': 0.99}
    ):
        self.numpy_rng = np.random.RandomState(1234)
        self.theano_rng = RandomStreams(self.numpy_rng.randint(2 ** 30))

        self.n_dim = n_dim
        self.n_out = n_out
        self.n_superbatch = n_superbatch
        self.alg = opt_alg
        self.n_class = 10

        lr = opt_params.get('lr')
        n_batch = opt_params.get('nb')

        train_set_x = theano.shared(
            np.empty(
                (n_superbatch, n_chan, n_dim, n_dim),
                dtype=theano.config.floatX
            ), borrow=False,
        )
        val_set_x = theano.shared(np.empty(
            (n_superbatch, n_chan, n_dim, n_dim),
            dtype=theano.config.floatX),
            borrow=False,
        )
        train_set_y = theano.shared(
            np.empty(
                (n_superbatch,),
                dtype=theano.config.floatX
            ), borrow=False,
        )
        val_set_y = theano.shared(
            np.empty(
                (n_superbatch,),
                dtype=theano.config.floatX
            ), borrow=False,
        )
        train_set_y_int = T.cast(train_set_y, 'int32')
        val_set_y_int = T.cast(val_set_y, 'int32')

        train_rbm_px_mu = theano.shared(
            np.empty(
                (n_superbatch, self.n_aux),
                dtype=theano.config.floatX
            ), borrow=False,
        )

        X = T.tensor4(dtype=theano.config.floatX)
        S = T.tensor3(dtype=theano.config.floatX)
        Y = T.ivector()
        px_mu = T.lscalar(dtype=config.floatX)
        idx1, idx2 = T.lscalar(), T.lscalar()
        alpha = T.scalar(dtype=theano.config.floatX)  # learning rate
        self.inputs = (X, Y, idx1, idx2, S, px_mu)

        # ----------------------------
        # Begin RBM-only
        self.rbm_network = self.create_rbm_model(n_dim, n_out, n_chan)
        persistent_chain = theano.shared(
            np.zeros(
                (n_batch, self.n_hidden),
                dtype=theano.config.floatX
            ), borrow=True,
        )
        rbm_cost, rbm_acc, rbm_updates = self.get_rbm_objective_and_updates(
            alpha, lr=lr, persistent=persistent_chain,
        )
        self.rbm_objectives = (rbm_cost, rbm_acc)
        self.rbm_train = theano.function(
            [idx1, idx2, alpha],
            [rbm_cost, rbm_acc],
            updates=rbm_updates,
            givens={
                X: train_set_x[idx1:idx2],
                Y: train_set_y_int[idx1:idx2]
            },
            on_unused_input='warn',
        )
        # End RBM-only
        # ----------------------------
        # Begin DADGM-only
        tau = theano.shared(
            np.float32(5.0), name='temperature',
            allow_downcast=True, borrow=False,
        )
        self.tau = tau
        self.dadgm_network = self.create_dadgm_model(
            X, Y, n_dim, n_out, n_chan,
        )
        dadgm_loss, dadgm_acc = self.create_dadgm_objectives(False)
        self.dadgm_objectives = (dadgm_loss, dadgm_acc)
        dadgm_params = self.get_dadgm_params()
        dadgm_grads = self.create_dadgm_gradients(dadgm_loss, False)
        dadgm_updates = self.create_dadgm_updates(
            dadgm_grads, dadgm_params, alpha, opt_alg, opt_params,
        )
        self.dadgm_train = theano.function(
            [idx1, idx2, alpha], [dadgm_loss, dadgm_acc],
            updates=dadgm_updates,
            givens={
                X: train_set_x[idx1:idx2],
                Y: train_set_y_int[idx1:idx2],
                px_mu: train_rbm_px_mu,
            },
            on_unused_input='warn',
        )
        self.dadgm_loss = theano.function(
            [X, Y], [dadgm_loss, dadgm_acc],
            on_unused_input='warn',
        )
        # End DADGM-only
        # ----------------------------
        self.n_batch = n_batch
        # parameters for sampling
        self.n_chain = 100

        # save data variables
        self.train_set_x = train_set_x
        self.train_set_y = train_set_y
        self.val_set_x = val_set_x
        self.val_set_y = val_set_y
        self.train_rbm_px_mu = train_rbm_px_mu
        self.data_loaded = False

    def create_rbm_model(self, n_dim, n_out, n_chan=1, n_class=10):
        n_visible = n_chan*n_dim*n_dim  # size of visible layer
        n_hidden = 500  # size of hidden layer
        k_steps = 15  # number of steps during CD/PCD

        initial_W = np.asarray(
            self.numpy_rng.uniform(
                low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                size=(n_visible, n_hidden)
            ), dtype=theano.config.floatX,
        )
        W = theano.shared(value=initial_W, name='W', borrow=True)
        hbias = theano.shared(
            value=np.zeros(
                n_hidden,
                dtype=theano.config.floatX
            ), name='hbias',
            borrow=True,
        )
        vbias = theano.shared(
            value=np.zeros(
                n_visible,
                dtype=theano.config.floatX
            ), name='vbias',
            borrow=True,
        )

        self.W = W
        self.hbias = hbias
        self.vbias = vbias

        self.rbm_params = [self.W, self.hbias, self.vbias]

        # network params
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k_steps = k_steps

        return None

    def create_dadgm_model(self, X, Y, n_dim, n_out, n_chan=1, n_class=10):
        n_cat = 20  # number of categorical distributions
        n_lat = n_class*n_cat  # latent stochastic variables
        n_aux = 10  # number of auxiliary variables
        n_hid = 500  # size of hidden layer in encoder/decoder
        n_in = n_out = n_dim * n_dim * n_chan
        tau = self.tau
        hid_nl = T.nnet.relu
        relu_shift = lambda av: T.nnet.relu(av+10)-10

        # create the encoder network
        # - create q(a|x)
        qa_net_in = InputLayer(shape=(None, n_in), input_var=X)
        qa_net = DenseLayer(
            qa_net_in, num_units=n_hid,
            W=GlorotNormal('relu'), b=Normal(1e-3),
            nonlinearity=hid_nl,
        )
        qa_net_mu = DenseLayer(
            qa_net, num_units=n_aux,
            W=GlorotNormal(), b=Normal(1e-3),
            nonlinearity=None,
        )
        qa_net_logsigma = DenseLayer(
            qa_net, num_units=n_aux,
            W=GlorotNormal(), b=Normal(1e-3),
            nonlinearity=relu_shift,
        )
        qa_net_sample = GaussianSampleLayer(qa_net_mu, qa_net_logsigma)
        # - create q(z|a, x)
        qz_net_in = lasagne.layers.InputLayer((None, n_aux))
        qz_net_a = DenseLayer(
            qz_net_in, num_units=n_hid,
            nonlinearity=hid_nl,
        )
        qz_net_b = DenseLayer(
            qa_net_in, num_units=n_hid,
            nonlinearity=hid_nl,
        )
        qz_net = ElemwiseSumLayer([qz_net_a, qz_net_b])
        qz_net = DenseLayer(
            qz_net, num_units=n_hid,
            nonlinearity=hid_nl
        )
        qz_net_mu = DenseLayer(
            qz_net, num_units=n_lat,
            nonlinearity=None,
        )
        qz_net_mu = reshape(qz_net_mu, (-1, n_class))
        qz_net_sample = GumbelSoftmaxSampleLayer(qz_net_mu, tau)
        qz_net_sample = reshape(qz_net_sample, (-1, n_cat, n_class))
        # create the decoder network
        # - create p(x|z)
        px_net_in = lasagne.layers.InputLayer((None, n_cat, n_class))
        # --- rest is created from RBM ---
        # - create p(a|z)
        pa_net = DenseLayer(
            flatten(px_net_in), num_units=n_hid,
            W=GlorotNormal('relu'), b=Normal(1e-3),
            nonlinearity=hid_nl,
        )
        pa_net_mu = DenseLayer(
            pa_net, num_units=n_aux,
            W=GlorotNormal(),
            b=Normal(1e-3),
            nonlinearity=None,
        )
        pa_net_logsigma = DenseLayer(
            pa_net, num_units=n_aux,
            W=GlorotNormal(),
            b=Normal(1e-3),
            nonlinearity=relu_shift,
        )
        # save network params
        self.n_cat = n_cat
        self.input_layers = (qa_net_in, qz_net_in, px_net_in)

        return pa_net_mu, pa_net_logsigma, qz_net_mu, \
            qa_net_mu, qa_net_logsigma, qz_net_sample, qa_net_sample,

    def get_rbm_objective_and_updates(self, alpha, lr=0.1, persistent=None):
        X = self.inputs[0]
        x = X.reshape((-1, self.n_visible))
        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(x)
        # for PCD, we initialize from the old state of the chain
        chain_start = persistent
        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=15,
            name="gibbs_hvh",
        )

        chain_end = nv_samples[-1]
        cost = T.mean(self.free_energy(x))-T.mean(self.free_energy(chain_end))
        # We must not compute the gradient through the gibbs sampling
        params = self.get_params()
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])
        gparams = [grad * alpha for grad in gparams]

        for gparam, param in zip(gparams, params):
            updates[param] = param - gparam * T.cast(lr, dtype=theano.config.floatX)

        updates[persistent] = nh_samples[-1]
        monitoring_cost = self.get_pseudo_likelihood_cost(x, updates)

        return monitoring_cost, monitoring_cost, updates

    def create_dadgm_objectives(self, deterministic=False):
        X = self.inputs[0]
        x = X.reshape((-1, self.n_out))
        px_mu = self.inputs[-1]

        # load network params
        n_class = self.n_class
        n_cat = self.n_cat

        pa_net_mu, pa_net_logsigma, qz_net_mu, qa_net_mu, \
            qa_net_logsigma, qz_net_sample, qa_net_sample = self.network
        qa_net_in, qz_net_in, px_net_in = self.input_layers

        qa_mu, qa_logsigma, qa_sample = get_output(
            [qa_net_mu, qa_net_logsigma, qa_net_sample],
            deterministic=deterministic,
        )
        qz_mu, qz_sample = get_output(
            [qz_net_mu, qz_net_sample],
            {qz_net_in: qa_sample, qa_net_in: x},
            deterministic=deterministic,
        )
        pa_mu, pa_logsigma = get_output(
            [pa_net_mu, pa_net_logsigma],
            {px_net_in: qz_sample},
            deterministic=deterministic,
        )
        # Load this from RBM 
        px_mu = self.inputs[-1]

        qz_given_ax = T.nnet.softmax(qz_mu)
        log_qz_given_ax = T.log(qz_given_ax + 1e-20)
        entropy = T.reshape(
            qz_given_ax * (log_qz_given_ax - T.log(1.0 / n_class)),
            (-1, n_cat, n_class),
        )
        entropy = T.sum(entropy, axis=[1, 2])

        log_px_given_z = log_bernoulli(x, px_mu).sum(axis=1)
        log_pa_given_z = log_normal2(qa_sample, pa_mu, pa_logsigma).sum(axis=1)
        log_paxz = log_pa_given_z + log_px_given_z

        # logp(z)+logp(a|z)-logq(a)-logq(z|a)
        elbo = T.mean(log_paxz - entropy)

        return -elbo, -T.mean(entropy)

    def create_dadgm_gradients(self, loss, deterministic=False):
        grads = GSM.create_gradients(self, loss, deterministic)

        # combine and clip gradients
        clip_grad, max_norm = 1, 5
        mgrads = total_norm_constraint(grads, max_norm=max_norm)
        cgrads = [T.clip(g, -clip_grad, clip_grad) for g in mgrads]

        return cgrads

    def create_dadgm_updates(self, grads, params, alpha, opt_alg, opt_params):
        return GSM.create_updates(
            self, grads, params, alpha, opt_alg, opt_params
        )

    def get_rbm_params(self):
        return self.rbm_params

    def get_dadgm_params(self):
        px_net_mu, pa_net_mu, pa_net_logsigma, \
            qz_net_mu, qa_net_mu, qa_net_logsigma, \
            qz_net_sample, qa_net_sample = self.dadgm_network

        p_params = get_all_params(
            [px_net_mu, pa_net_mu, pa_net_logsigma],
            trainable=True,
        )
        qa_params = get_all_params(qa_net_sample, trainable=True)
        qz_params = get_all_params(qz_net_sample, trainable=True)

        return p_params + qa_params + qz_params

    def fit_dadgm(
        self, X_train, Y_train, X_val, Y_val,
        n_epoch=10, n_batch=100, logname='run',
    ):
        alpha = 1.0  # learning rate, which can be adjusted later
        tau0 = 1.0  # initial temp
        MIN_TEMP = 0.5  # minimum temp
        ANNEAL_RATE = 0.00003  # adjusting rate
        np_temp = tau0
        n_data = len(X_train)
        n_superbatch = self.n_superbatch
        i = 1  # track # of train() executions

        n_flat_dim = np.prod(X_train.shape[1:])
        X_train = X_train.reshape(-1, n_flat_dim)
        X_val = X_val.reshape(-1, n_flat_dim)

        for epoch in range(n_epoch):
            # In each epoch, we do a full pass over the training data:
            train_batches, train_err, train_acc = 0, 0, 0
            start_time = time.time()

            # iterate over superbatches to save time on GPU memory transfer
            for X_sb, Y_sb in self.iterate_superbatches(
                X_train, Y_train, n_superbatch,
                datatype='train', shuffle=True,
            ):
                for idx1, idx2 in iterate_minibatch_idx(len(X_sb), n_batch):
                    dadgm_err, dadgm_acc = self.dadgm_train(idx1, idx2, alpha)

                    # anneal temp and learning rate
                    if i % 1000 == 1:
                        alpha *= 0.9
                        np_temp = np.maximum(tau0*np.exp(-ANNEAL_RATE*i), MIN_TEMP)
                        self.tau.set_value(np_temp, borrow=False)

                    # collect metrics
                    i += 1
                    train_batches += 1
                    train_err += err
                    train_acc += acc

    def fit(
        self, X_train, Y_train, X_val, Y_val,
        n_epoch=10, n_batch=100, logname='run'
    ):
        """Train 1 epoch of PCD to learn p, followed by K epochs
        of DADGM via GSM to fit q to p.
        """
        alpha = 1.0  # learning rate, which can be adjusted later
        n_data = len(X_train)
        n_superbatch = self.n_superbatch
        n_loop_q_per_p = 10

        n_flat_dim = np.prod(X_train.shape[1:])
        X_train = X_train.reshape(-1, n_flat_dim)
        X_val = X_val.reshape(-1, n_flat_dim)

        for epoch in range(n_epoch):
            # In each epoch, we do a full pass over the training data:
            train_batches, train_err, train_acc = 0, 0, 0
            start_time = time.time()

            # iterate over superbatches to save time on GPU memory transfer
            for X_sb, Y_sb in self.iterate_superbatches(
                X_train, Y_train, n_superbatch,
                datatype='train', shuffle=True,
            ):
                for idx1, idx2 in iterate_minibatch_idx(len(X_sb), n_batch):
                    rbm_err, rbm_acc = self.rbm_train(idx1, idx2, alpha)
                    self.train_rbm_px_mu.set_value(rbm_err)
                    # Fit n_loop epochs of fitting q to current p
                    self.fit_dadgm(
                        X_train, Y_train, X_val, Y_val,
                        n_epoch=n_loop_q_per_p, n_batch=n_batch, logname=logname,
                    )

                    train_batches += 1
                    train_err += err
                    train_acc += acc

    def hallucinate(self):
        pass

    def load_params(self, rbm_params, dadgm_params):
        """Load a given set of parameters"""
        self.rbm_params = rbm_params
        self.dadgm_params = dadgm_params

    def dump_params(self):
        """Dump a given set of parameters"""
        return (self.rbm_params, self.dadgm_params)
