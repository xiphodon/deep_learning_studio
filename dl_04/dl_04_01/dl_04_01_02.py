#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/29 16:17
# @Author  : GuoChang
# @Site    : https://github.com/xiphodon
# @File    : dl_04_01_02.py
# @Software: PyCharm

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from dl_04.dl_04_01.cnn_utils import *

np.random.seed(1)


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes

    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    ### START CODE HERE ### (≈2 lines)
    X = tf.placeholder(dtype=tf.float32, shape=(None, n_H0, n_W0, n_C0), name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=(None, n_y), name='Y')
    ### END CODE HERE ###

    return X, Y


def create_placeholders_test():
    """
    create_placeholders 测试
    :return:
    """
    X, Y = create_placeholders(64, 64, 3, 6)
    print("X = " + str(X))
    print("Y = " + str(Y))


def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """

    tf.set_random_seed(1)  # so that your "random" numbers match ours

    ### START CODE HERE ### (approx. 2 lines of code)
    W1 = tf.get_variable(name='W1', shape=(4, 4, 3, 8), initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable(name='W2', shape=(2, 2, 8, 16), initializer=tf.contrib.layers.xavier_initializer(seed=0))
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "W2": W2}

    return parameters


def initialize_parameters_test():
    """
    initialize_parameters 测试
    :return:
    """
    tf.reset_default_graph()
    with tf.Session() as sess_test:
        parameters = initialize_parameters()
        init = tf.global_variables_initializer()
        sess_test.run(init)
        print("W1 = " + str(parameters["W1"].eval()[1, 1, 1]))
        print("W2 = " + str(parameters["W2"].eval()[1, 1, 1]))


def start():
    """
    入口
    :return:
    """
    # Loading the data (signs)
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    # Example of a picture
    index = 6
    plt.imshow(X_train_orig[index])
    print("y = " + str(np.squeeze(Y_train_orig[:, index])))
    plt.show()

    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T
    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))
    conv_layers = {}


if __name__ == '__main__':
    # start()
    # create_placeholders_test()
    initialize_parameters_test()
