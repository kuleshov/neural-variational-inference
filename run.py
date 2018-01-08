import pdb
import argparse
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from util import data

# ----------------------------------------------------------------------------

def make_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Commands')

    # train
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.set_defaults(func=train)

    train_parser.add_argument('--dataset', default='mnist')
    train_parser.add_argument('--model', default='vae')
    train_parser.add_argument('--pkl')
    train_parser.add_argument('-e', '--epochs', type=int, default=10)
    train_parser.add_argument('-l', '--logname', default='mnist-run')
    train_parser.add_argument('-p', '--plotname', default='mnist-plot.png')
    train_parser.add_argument('--alg', default='adam')
    train_parser.add_argument('--lr', type=float, default=1e-3)
    train_parser.add_argument('--b1', type=float, default=0.9)
    train_parser.add_argument('--b2', type=float, default=0.999)
    train_parser.add_argument('--n_batch', type=int, default=128)
    train_parser.add_argument('--n_superbatch', type=int, default=1280)

    return parser

# ----------------------------------------------------------------------------

def train(args):
    import models
    import numpy as np
    np.random.seed(1234)

    if args.dataset == 'mnist':
        n_dim, n_out, n_channels = 28, 10, 1
        X_train, Y_train, X_val, Y_val, _, _ = data.load_mnist()
    elif args.dataset == 'binmnist':
        n_dim, n_out, n_channels = 28, 10, 1
        X_train, X_val, _ = data.load_mnist_binarized()
        X_train = X_train.reshape((-1, 1, 28, 28))
        X_val = X_val.reshape((-1, 1, 28, 28))
        Y_train = np.empty((X_train.shape[0],), dtype='int32')
        Y_val = np.empty((X_val.shape[0],), dtype='int32')
    elif args.dataset == 'omniglot':
        n_dim, n_out, n_channels = 28, 10, 1
        X_train, Y_train, X_val, Y_val = data.load_omniglot_iwae()
        X_train = X_train.reshape((-1, 1, 28, 28))
        X_val = X_val.reshape((-1, 1, 28, 28))
        Y_train = np.empty((X_train.shape[0],), dtype='int32')
        Y_val = np.empty((X_val.shape[0],), dtype='int32')
    elif args.dataset == 'digits':
        n_dim, n_out, n_channels = 8, 10, 1
        X_train, Y_train, X_val, Y_val, _, _ = data.load_digits()
    else:
        X_train, Y_train = data.load_h5(args.train)
        X_val, Y_val = data.load_h5(args.test)

    # also get the data dimensions
    print 'dataset loaded.'

    # set up optimization params
    p = { 'lr' : args.lr, 'b1': args.b1, 'b2': args.b2, 'nb': args.n_batch }

    # create model
    if args.model == 'vae':
        model = models.VAE(
            n_dim=n_dim, n_out=n_out, n_chan=n_channels,
            n_superbatch=args.n_superbatch, opt_alg=args.alg, opt_params=p,
        )
    elif args.model == 'discrete-vae':
        model = models.SBN(
            n_dim=n_dim, n_out=n_out, n_chan=n_channels,
            n_superbatch=args.n_superbatch, opt_alg=args.alg, opt_params=p,
        )
    elif args.model == 'discrete-vae-rbm':
        model = models.USBN(
            n_dim=n_dim, n_out=n_out, n_chan=n_channels,
            n_superbatch=args.n_superbatch, opt_alg=args.alg, opt_params=p,
        )
    elif args.model == 'adgm':
        model = models.ADGM(
            n_dim=n_dim, n_out=n_out, n_chan=n_channels,
            n_superbatch=args.n_superbatch, opt_alg=args.alg, opt_params=p,
        )
    elif args.model == 'discrete-adgm':
        model = models.DADGM(
            n_dim=n_dim, n_out=n_out, n_chan=n_channels,
            n_superbatch=args.n_superbatch, opt_alg=args.alg, opt_params=p,
        )
    elif args.model == 'discrete-adgm-rbm':
        model = models.UDADGM(
            n_dim=n_dim, n_out=n_out, n_chan=n_channels,
            n_superbatch=args.n_superbatch, opt_alg=args.alg, opt_params=p,
        )
    elif args.model == 'rbm':
        model = models.RBM(
            n_dim=n_dim, n_out=n_out, n_chan=n_channels,
            n_superbatch=args.n_superbatch, opt_alg=args.alg, opt_params=p,
        )
    elif args.model == 'vrbm':
        model = models.VariationalRBM(
            n_dim=n_dim, n_out=n_out, n_chan=n_channels,
            n_superbatch=args.n_superbatch, opt_alg=args.alg, opt_params=p,
        )
    elif args.model == 'avrbm':
        model = models.AuxiliaryVariationalRBM(
            n_dim=n_dim, n_out=n_out, n_chan=n_channels,
            n_superbatch=args.n_superbatch, opt_alg=args.alg, opt_params=p,
        )
    else:
        raise ValueError('Invalid model')

    if args.pkl:
        model.load(args.pkl)
        
        # generate samples
        samples = model.hallucinate_chain()

        # plot them
        plt.figure(figsize=(5, 5))
        plt.imshow(samples, cmap=plt.cm.gray, interpolation='none')
        plt.title('Hallucinated Samples')
        plt.tight_layout()
        plt.savefig(args.plotname)

        exit('hello')

    # train model
    model.fit(
        X_train, Y_train, X_val, Y_val,
        n_epoch=args.epochs, n_batch=args.n_batch,
        logname=args.logname
    )

    # generate samples
    samples = model.hallucinate()

    # plot them
    plt.figure(figsize=(5, 5))
    plt.imshow(samples, cmap=plt.cm.gray, interpolation='none')
    plt.title('Hallucinated Samples')
    plt.tight_layout()
    plt.savefig(args.plotname)

def main():
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
