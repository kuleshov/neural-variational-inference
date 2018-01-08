import sys
import os
import pickle
import tarfile
import numpy as np
import urllib
import zipfile
import fnmatch
import shutil
import gzip
import cPickle as cPkl
import pickle as pkl


def whiten(X_train, X_valid):
    offset = np.mean(X_train, 0)
    scale = np.std(X_train, 0).clip(min=1)
    X_train = (X_train - offset) / scale
    X_valid = (X_valid - offset) / scale
    return X_train, X_valid


def load_cifar10():
    """Download and extract the tarball from Alex's website."""
    dest_directory = '.'
    DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        if sys.version_info[0] == 2:
            from urllib import urlretrieve
        else:
            from urllib.request import urlretrieve

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    def load_CIFAR_batch(filename):
        """ load single batch of cifar """
        with open(filename, 'rb') as f:
            datadict = pickle.load(f)
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000, 3, 32, 32).astype("float32")
            Y = np.array(Y, dtype=np.uint8)
            return X, Y

    xs, ys = [], []
    for b in range(1,6):
        f = 'cifar-10-batches-py/data_batch_%d' % b
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)

    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch('cifar-10-batches-py/test_batch')

    return Xtr, Ytr, Xte, Yte

def load_mnist():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print "Downloading %s" % filename
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)

        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)

        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test

def _download_omniglot_iwae(dataset):
    """
    Download the MNIST dataset if it is not present.
    :return: The train, test and validation set.
    """
    origin = (
        'https://github.com/yburda/iwae/raw/'
        'master/datasets/OMNIGLOT/chardata.mat'
    )
    print 'Downloading data from %s' % origin
    urllib.urlretrieve(origin, dataset + '/chardata.mat')

def load_omniglot_iwae(dataset='./omniglot_iwae'):
    '''
    Loads the real valued MNIST dataset
    :param dataset: path to dataset file
    :return: None
    '''
    from scipy.io import loadmat
    if not os.path.exists(dataset):
        os.makedirs(dataset)
        _download_omniglot_iwae(dataset)

    data = loadmat(dataset+'/chardata.mat')

    train_x = data['data'].astype('float32').T
    train_t = np.argmax(data['target'].astype('float32').T,axis=1)
    train_char = data['targetchar'].astype('float32')
    test_x = data['testdata'].astype('float32').T
    test_t = np.argmax(data['testtarget'].astype('float32').T,axis=1)
    test_char = data['testtargetchar'].astype('float32')


    return train_x, train_t, test_x, test_t

def _download_mnist_binarized(datapath):
    """
    Download the fized binzarized MNIST dataset if it is not present.
    :return: The train, test and validation set.
    """
    datafiles = {
        "train": "http://www.cs.toronto.edu/~larocheh/public/"
                 "datasets/binarized_mnist/binarized_mnist_train.amat",
        "valid": "http://www.cs.toronto.edu/~larocheh/public/datasets/"
                 "binarized_mnist/binarized_mnist_valid.amat",
        "test": "http://www.cs.toronto.edu/~larocheh/public/datasets/"
                "binarized_mnist/binarized_mnist_test.amat"
    }
    datasplits = {}
    for split in datafiles.keys():
        print "Downloading %s data..." %(split)
        local_file = datapath + '/binarized_mnist_%s.npy'%(split)
        datasplits[split] = np.loadtxt(urllib.urlretrieve(datafiles[split])[0])

    f = gzip.open(datapath +'/mnist.pkl.gz', 'w')
    pkl.dump([datasplits['train'],datasplits['valid'],datasplits['test']],f)

# def load_mnist_binarized(dataset='./mnist_binarized/mnist.pkl.gz'):
def load_mnist_binarized(dataset='./mnist_binarized/mnist.pkl'):
    '''
    Loads the fixed binarized MNIST dataset provided by Hugo Larochelle.
    :param dataset: path to dataset file
    :return: None
    '''
    if not os.path.isfile(dataset):
        datasetfolder = os.path.dirname(dataset)
        if not os.path.exists(datasetfolder):
            os.makedirs(datasetfolder)
        _download_mnist_binarized(datasetfolder)

    # f = gzip.open(dataset, 'rb')
    f = open(dataset, 'rb')
    x_train, x_valid, x_test = pkl.load(f)
    f.close()

    x_train = x_train.astype('float32')
    x_valid = x_valid.astype('float32')
    x_test = x_test.astype('float32')

    return x_train, x_valid, x_test    

def load_digits():
    from sklearn.datasets import load_digits as _load_digits
    from sklearn.cross_validation import train_test_split

    digits = _load_digits()
    X = np.asarray(digits.data, 'float32')
    X, Y = nudge_dataset(X, digits.target)
    X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling
    n, d2 = X.shape
    d = int(np.sqrt(d2))
    X = X.reshape((n,1,d,d))
    Y = np.array(Y, dtype=np.uint8)

    X_train, X_val, y_train, y_val = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=0)
    # TODO: We don't use these right now
    X_test, y_test = None, None

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test

# def load_digits():
#     from sklearn.datasets import load_digits as _load_digits
#     data = _load_digits()
#     X, y = data.images.astype('float32'), data.target.astype('int32')
#     X = X[:, np.newaxis, :, :]

#     # We reserve the last  300 / ~1800 for validation.
#     X_train, X_val = X[:-300], X[-300:]
#     y_train, y_val = y[:-300], y[-300:]

#     # TODO: We don't use these right now
#     X_test, y_test = None, None

#     # We just return all the arrays in order, as expected in main().
#     # (It doesn't matter how we do this as long as we can read them again.)
#     return X_train, y_train, X_val, y_val, X_test, y_test


def split_semisup(X, y, n_lbl):
    n_tot = len(X)
    idx = np.random.permutation(n_tot)

    X_lbl = X[idx[:n_lbl]].copy()
    X_unl = X[idx[n_lbl:]].copy()
    y_lbl = y[idx[:n_lbl]].copy()
    y_unl = y[idx[n_lbl:]].copy()

    return X_lbl, y_lbl, X_unl, y_unl


def load_noise(n=100,d=5):
    """For debugging"""
    X = np.random.randint(2,size=(n,1,d,d)).astype('float32')
    Y = np.random.randint(2,size=(n,)).astype(np.uint8)

    return X, Y


def load_h5(h5_path):
    """This was untested"""
    import h5py
    # load training data
    with h5py.File(h5_path, 'r') as hf:
        print 'List of arrays in input file:', hf.keys()
        X = np.array(hf.get('data'))
        Y = np.array(hf.get('label'))
        print 'Shape of X: \n', X.shape
        print 'Shape of Y: \n', Y.shape

        return X, Y


def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    from scipy.ndimage import convolve
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y
