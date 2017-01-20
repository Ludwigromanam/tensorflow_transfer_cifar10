import cPickle as pickle
import numpy as np
import os
from extract import batch_pool3_features, create_graph_session

# Make sure to download and extract CIFAR-10 data before
# running this (https://www.cs.toronto.edu/~kriz/cifar.html)

CIFAR10_DIR = './cifar-10-batches-py'

def load_CIFAR_batch(file_loc):
    """
    load a single batch of CIFAR data

    Parameters
    ---------
    file_loc : location of batch

    Returns
    -------
    X is our batch inputs
    Y is our batch labels
    """
    with open(file_loc, 'r') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        # split into rgb 32x32 image. Transpose so that channel is final dimension.
        X = X.reshape(X.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
        # cast as float
        X = X.astype("float")
        Y = np.array(Y)
    return X, Y

def load_CIFAR10():
    """
    load all of cifar

    Returns
    -------
    training and testing inputs and labels
    """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(CIFAR10_DIR, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_test()
    return Xtr, Ytr, Xte, Yte

def load_CIFAR_test():
    """
    load only the CIFAR test images

    Returns
    -------
    testing inputs and labels
    """
    X_test, Y_test = load_CIFAR_batch(os.path.join(CIFAR10_DIR, 'test_batch'))
    return X_test, Y_test

def load_pool3_outputs():
    """
    load our images and labels

    Returns
    -------
    images and labels split by train and test
    """
    X_test_file = 'X_test.npy'
    y_test_file = 'y_test.npy'
    X_train_file = 'X_train.npy'
    y_train_file = 'y_train.npy'
    return np.load(X_train_file), np.load(y_train_file), np.load(X_test_file), np.load(y_test_file)

def cifar_pool3_outputs(X,filename):
    """
    save pool3 outputs to a file so we don't need to recalculate

    Parameters
    ----------
    X : images
    filename : location to store outputs from pool3 layer

    Returns
    -------
    Nothing
    """
    print 'About to generate file: %s' % filename
    sess = create_graph_session()
    X_pool3 = batch_pool3_features(sess, X)
    np.save(filename, X_pool3)

def serialize_data():
    """
    send our images through inception-v3 network and store outputs
    from second to last layer(bottleneck layer). This is useful b/c it saves
    us a lot of time when tuning our network since these values are always
    the same.

    Returns
    -------
    Nothing
    """
    X_train, y_train, X_test, y_test = load_CIFAR10()
    cifar_pool3_outputs(X_train, 'X_train')
    cifar_pool3_outputs(X_test, 'X_test')
    np.save('y_train', y_train)
    np.save('y_test', y_test)

