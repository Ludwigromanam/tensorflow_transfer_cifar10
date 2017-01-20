from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np

MODEL = './classify_image_graph_def.pb'

def create_graph_session():
    """"
    Creates a graph from saved GraphDef file and returns a saver.

    Returns
    -------
    tensor flow session
    """
    # Creates graph from saved graph_def.pb.
    print 'Loading graph...'
    with tf.Session() as sess:
        with gfile.FastGFile(MODEL, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    return sess


def batch_pool3_features(sess, X_input):
    """
    Currently tensorflow can't extract pool3 in batch so this is slow:
    https://github.com/tensorflow/tensorflow/issues/1021

    Parameters
    ----------
    sess : tf session
    X_input : inputs

    Returns
    -------
    outputs from pool 3 layer
    """
    n_train = X_input.shape[0]
    print 'Extracting features for %i rows' % n_train
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    X_pool3 = []
    for i in range(n_train):
        print 'Iteration %i' % i
        pool3_features = sess.run(pool3,{'DecodeJpeg:0': X_input[i,:]})
        X_pool3.append(np.squeeze(pool3_features))
    return np.array(X_pool3)

def iterate_mini_batches(X_input,Y_input,batch_size):
    """

    Parameters
    ----------
    X_input : inputs
    Y_input: labels
    batch_size: size of mini batches

    Returns
    -------
    inputs and outputs to iterate over for training
    """

    n_train = X_input.shape[0]
    for ndx in range(0, n_train, batch_size):
        yield X_input[ndx:min(ndx + batch_size, n_train)], Y_input[ndx:min(ndx + batch_size, n_train)]