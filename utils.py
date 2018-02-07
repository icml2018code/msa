""" Utility functions """

import pickle
import numpy as np
import tensorflow as tf
import itertools


def smooth_hinge_loss(labels, logits):
    """
    Square smoothed hinge loss
    """
    labels = 2*labels - 1  # make it +1 and -1
    loss = tf.nn.relu(1 - logits*labels)
    loss_mean_squared = tf.reduce_mean(loss**2)
    return loss_mean_squared


def get_ph(tensor):
    """
    Helper function that returns a tf.placeholder
    tensor that is the same size/dtype as the input
    tensor
    """
    return tf.placeholder(tensor.dtype, tensor.get_shape())


def get_sparsity_frac(network, trainer):
    """
    Returns the fraction of non-zero weights
    """
    num_params = np.sum(
        [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    disc_vars = [v.vars[0] for v in network.layers if v.is_discrete]
    disc_vars_values = trainer.sess.run(disc_vars)
    flattened = [v.flatten() for v in disc_vars_values]
    joined = np.concatenate(flattened)
    sparsity_frac = np.sum(np.abs(joined) > 0)/num_params
    print('Sparsity fraction (ratio of non-zero weights): ', sparsity_frac)
    return sparsity_frac


def load_mnist_data():
    """ Load MNIST dataset """
    # load MNIST dataset
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    images = np.concatenate(
        [mnist.train.images,
         mnist.validation.images,
         mnist.test.images], axis=0)
    labels = np.concatenate(
        [mnist.train.labels,
         mnist.validation.labels,
         mnist.test.labels], axis=0)

    n_train, n_valid, n_test = 55000, 5000, 10000

    x_train = images[:n_train]
    y_train = labels[:n_train]
    x_validate = images[n_train:n_train+n_valid]
    y_validate = labels[n_train:n_train+n_valid]
    x_test = images[-n_test:]
    y_test = labels[-n_test:]
    
    return x_train, y_train, x_validate, y_validate, x_test, y_test


def load_cifar10_data():
    """ Load CIFAR-10 dataset """
    import cifar10
    cifar10.maybe_download_and_extract()
    images_train, cls_train, labels_train = cifar10.load_training_data()
    images_test, cls_test, labels_test = cifar10.load_test_data()

    x_train = images_train[:45000]
    y_train = labels_train[:45000]

    x_validate = images_train[45000:]
    y_validate = labels_train[45000:]

    x_test = images_test
    y_test = labels_test
    
    return x_train, y_train, x_validate, y_validate, x_test, y_test
    
"""
Converting Labels to One Hot Encoding and Image Matrix to favourable dimensions
"""
def reformat(data, Y):
    xtrain = []
    trainLen = data.shape[3]
    for x in range(trainLen):
        xtrain.append(data[:,:,:,x])
    xtrain = np.asarray(xtrain)
    Ytr=[]
    for el in Y:
        temp=np.zeros(10)
        if el==10:
            temp[0]=1
        elif el==1:
            temp[1]=1
        elif el==2:
            temp[2]=1
        elif el==3:
            temp[3]=1
        elif el==4:
            temp[4]=1
        elif el==5:
            temp[5]=1
        elif el==6:
            temp[6]=1
        elif el==7:
            temp[7]=1
        elif el==8:
            temp[8]=1
        elif el==9:
            temp[9]=1
        Ytr.append(temp)
    return xtrain, np.asarray(Ytr)

def load_svhn_data():
    """ Load SVHN dataset """
    import scipy.io
    x_train = scipy.io.loadmat('./data/svhn/train_32x32.mat')['X']
    y_train = scipy.io.loadmat('./data/svhn/train_32x32.mat')['y']
    x_test = scipy.io.loadmat('./data/svhn/test_32x32.mat')['X']
    y_test = scipy.io.loadmat('./data/svhn/test_32x32.mat')['y']
    
    x_train, y_train = reformat(x_train, y_train)
    x_test, y_test = reformat(x_test, y_test)

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_validate = x_train[-5000:]
    y_validate = y_train[-5000:]

    x_train = x_train[:-5000]
    y_train = y_train[:-5000]
    
    return x_train, y_train, x_validate, y_validate, x_test, y_test
    
