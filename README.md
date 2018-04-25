Neural Variational Inference in Undirected Models
=================================================

This repository contains code accompanying the paper

```
Neural variational inference and learning in undirected graphical models.
Volodymyr Kuleshov and Stefano Ermon.
Neural Information Processing Systems, 2017
```

## Installation

The code uses Theano and Lasagne.
To install this package, clone the git repo, and update your `PYTHONPATH` to include the folder.

```
git clone https://github.com/kuleshov/neural-variational-inference.git;
cd neural-variational-inference/;
source set_env.sh
```

## Models and Datasets

The repository implements the following models:

* `vae`: Regular variational autoencoder.
* `discrete-vae`: Variational autoencoder with a `Bernoulli(200)` prior.
* `discrete-vae-rbm`: Variational autoencoder with an `RBM(64,8)` prior, 
   trained using neural variational inference.
* `adgm` Auxiliary-variable deep generative model (ADGM; Malloe et al. 2016).
* `discrete-adgm`: ADGM with a `Bernoulli(200)` prior.
* `discrete-adgm-rbm`: ADGM with an `RBM(64,8)` prior,·
   trained using neural variational inference.
* `rbm`: Regular Restricted Boltzmann Machine (RBM) trained with persistent contrastive divergence.
* `vrbm`: Variational RBM, i.e. RBM trained with neural variational infernce using a mixture of ten Bernoullis as the auxiliary helper distribution `q`.
* `avrbm`: Auxiliary-variable Variational RBM, i.e. RBM trained with neural variational infernce using an auxiliary-variable distribution `q(x,a)` (parametrized with a neural network) as the helper distribution `q`.

The models can be run on the following datasets:

* `digits`: The UCI digits dataset. Use this for the RBM models (otherwise you'll get numerical issues and will 
* `mnist`: Regular MNIST.
* `binmnist`: Binarized MNIST, using the binarization from the IWAE (Burda et al.) paper.
* `omniglot`: The Omniglot dataset.

The `run.py` script takes these names as input.

## Running the Code

To run a model, you may use the `run.py` launch script.

```
usage: run.py train [-h] [--dataset DATASET] [--model MODEL] [--pkl PKL]
                    [-e EPOCHS] [-l LOGNAME] [-p PLOTNAME] [--alg ALG]
                    [--lr LR] [--b1 B1] [--b2 B2] [--n_batch N_BATCH]
                    [--n_superbatch N_SUPERBATCH]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET
  --model MODEL
  --pkl PKL
  -e EPOCHS, --epochs EPOCHS
  -l LOGNAME, --logname LOGNAME
  -p PLOTNAME, --plotname PLOTNAME
  --alg ALG
  --lr LR
  --b1 B1
  --b2 B2
  --n_batch N_BATCH
  --n_superbatch N_SUPERBATCH
```

The simplest way to use it is via the `Makefile` provided in the root dir; typing `make train` will start training. 
You can specify the model, dataset, and other parameters by modifying the defaults in the `Makefile`.

The default hyper-parameters on `avrbm` on the `digits` dataset are currently set incorrectly and are causing problems.

## Feedback

Send feedback to [Volodymyr Kuleshov](http://www.stanford.edu/~kuleshov).
