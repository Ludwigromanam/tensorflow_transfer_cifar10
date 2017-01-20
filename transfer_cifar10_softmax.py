"""
Trying out the transfer learning example from:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py
"""


import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from data_utils import load_pool3_outputs, load_CIFAR_test
from extract import create_graph_session, iterate_mini_batches
import matplotlib.pyplot as plt
from tsne import tsne
#import seaborn as sns
import pandas as pd


CLASSES = np.array(['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
                    'horse', 'ship', 'truck'])
X_train_pool3, y_train_pool3, X_test_pool3, y_test_pool3 = load_pool3_outputs()
X_train, X_validation, Y_train, y_validation = train_test_split(X_train_pool3,
                            y_train_pool3, test_size=0.20)

GRAPH = create_graph_session().graph
bottleneck_tensor = GRAPH.get_tensor_by_name('pool_3/_reshape:0')
BOTTLENECK_TENSOR_SIZE = bottleneck_tensor._shape[1].value
# allows for larger batch sizes as opposed to just 1 if we used bottleneck
# tensor
BOTTLENECK_INPUT = tf.placeholder(tf.float32,
                                  shape=[None, BOTTLENECK_TENSOR_SIZE],
                                  name='BottleneckInputPlaceholder')

GROUND_TRUTH_TENSOR = tf.placeholder(tf.float32,
                                     [None, len(CLASSES)],
                                     name='gound_truth')

TRAINING_STEPS = 100
LEARNING_RATE = 0.01
FINAL_TENSOR_NAME = 'softmax_output'
PRINT_PROGRESS_INTERVAL = 10


# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py
def ensure_name_has_port(tensor_name):
    """Makes sure that there's a port number at the end of the tensor name.

    Parameters
    ----------
    tensor_name : A string representing the name of a tensor in a graph.

    Returns
    -------
    The input string with a :0 appended if no port was specified.
    """
    if ':' not in tensor_name:
        name_with_port = tensor_name + ':0'
    else:
        name_with_port = tensor_name
    return name_with_port

# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py
def add_final_training_ops():
    """Adds a new softmax and fully-connected layer for training.
    We need to retrain the top layer to identify our new classes, so this
    function adds the right operations to the graph, along with some variables
    to hold the weights, and then sets up all the gradients for the backward
    pass. The set up for the softmax and fully-connected layers is based on:
    https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

    Returns
    -------
    train_step, cross_entropy_mean
    """
    # get fully connected final layer weights
    layer_weights = tf.Variable(
        tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, len(CLASSES)],
                            stddev=0.001), name='final_weights')
    # make biases
    layer_biases = tf.Variable(tf.zeros([len(CLASSES)]), name='final_biases')

    logits = tf.matmul(BOTTLENECK_INPUT, layer_weights,
                       name='final_matmul') + layer_biases

    tf.nn.softmax(logits, name=FINAL_TENSOR_NAME)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits, GROUND_TRUTH_TENSOR)

    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(
        cross_entropy_mean)
    return train_step, cross_entropy_mean

# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py
def add_evaluation_step():
    """Inserts the operations we need to evaluate the accuracy of our results.

    Returns
    -------
    evaluation step
    """
    result_tensor = GRAPH.get_tensor_by_name(ensure_name_has_port(
        FINAL_TENSOR_NAME))
    correct_prediction = tf.equal(
        tf.argmax(result_tensor, 1), tf.argmax(GROUND_TRUTH_TENSOR, 1))
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    return evaluation_step

def encode_one_hot(y):
    """Inserts the operations we need to evaluate the accuracy of our results.

    Parameters
    ----------
    y : labels we need ot encode

    Returns
    -------
    evaluation_step
    """
    return np.eye(len(CLASSES))[y]

def do_train(sess,X_input, Y_input, X_validation, Y_validation):
    """
    training and validation for our network

    Parameters
    ----------
    sess : tensorflow session variable
    X_input : training inputs
    Y_input : training labels
    X_validation : validation inputs
    Y_validation : validation labels

    Returns
    -------
    final test accuracy
    """
    mini_batch_size = 10
    n_train = X_input.shape[0]

    train_step, cross_entropy = add_final_training_ops()

    init = tf.initialize_all_variables()
    sess.run(init)

    evaluation_step = add_evaluation_step()

    i = 0
    epocs = 1
    for epoch in range(epocs):
        shuffledRange = np.random.permutation(n_train)
        y_one_hot_train = encode_one_hot(Y_input)
        y_one_hot_validation = encode_one_hot(Y_validation)
        shuffledX = X_input[shuffledRange, :]
        shuffledY = y_one_hot_train[shuffledRange]
        for Xi, Yi in iterate_mini_batches(shuffledX, shuffledY,
                                           mini_batch_size):
            sess.run(train_step,
                     feed_dict={BOTTLENECK_INPUT: Xi,
                                GROUND_TRUTH_TENSOR: Yi})
            # Every so often, print out how well the graph is training.
            is_last_step = (i + 1 == TRAINING_STEPS)
            if (i % PRINT_PROGRESS_INTERVAL) == 0 or is_last_step:
                train_accuracy, cross_entropy_value = sess.run(
                  [evaluation_step, cross_entropy],
                  feed_dict={BOTTLENECK_INPUT: Xi,
                             GROUND_TRUTH_TENSOR: Yi})
                validation_accuracy = sess.run(
                  evaluation_step,
                  feed_dict={BOTTLENECK_INPUT: X_validation,
                             GROUND_TRUTH_TENSOR: y_one_hot_validation})
                print('Step %d: Train accuracy = %.1f%%, Cross entropy = %f,'
                      ' Validation accuracy = %.1f%%' %
                    (i, train_accuracy * 100, cross_entropy_value,
                     validation_accuracy * 100))
            i += 1

    test_accuracy = sess.run(
        evaluation_step,
        feed_dict={BOTTLENECK_INPUT: X_test_pool3,
                   GROUND_TRUTH_TENSOR: encode_one_hot(y_test_pool3)})
    print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

def show_test_images(sess):
    X_img, Y = load_CIFAR_test()
    n = X_img.shape[0]

    def rand_ordering():
        return np.random.permutation(n)

    def sequential_ordering():
        return range(n)

    for i in sequential_ordering():
        Xi_img = X_img[i, :]
        Xi_features = X_test_pool3[i, :].reshape(1, BOTTLENECK_TENSOR_SIZE)

        result_tensor = sess.graph.get_tensor_by_name(
            ensure_name_has_port(FINAL_TENSOR_NAME))

        probs = sess.run(result_tensor,
                         feed_dict={BOTTLENECK_INPUT: Xi_features})
        predicted_class = CLASSES[np.argmax(probs)]
        Yi = Y[i]
        Yi_label = CLASSES[Yi]
        plt.title('true=%s, predicted=%s' % (Yi_label, predicted_class))
        plt.imshow(Xi_img.astype('uint8'))
        print Yi_label
        plt.show()
        plt.close()


def plot_tsne(X_input_pool3, Y_input,n=10000):
    indicies = np.random.permutation(X_input_pool3.shape[0])[0:n]
    Y = tsne(X_input_pool3[indicies])
    num_labels = Y_input[indicies]
    labels = CLASSES[num_labels]
    df = pd.DataFrame(np.column_stack((Y,num_labels,labels)),
                      columns=["x1","x2","y","y_label"])
    sns.lmplot("x1", "x2", data=df.convert_objects(convert_numeric=True),
               hue="y_label", fit_reg=False, legend=True, palette="Set1")
    print 'done'

# plot_tsne(X_train_pool3 ,y_train_pool3)
sess = tf.InteractiveSession()
do_train(sess, X_train, Y_train, X_validation, y_validation)
show_test_images(sess)


