#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/15 11:13
# @Author  : GuoChang
# @Site    : https://github.com/xiphodon
# @File    : dl_01_02.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import lr_utils


def show_example_of_a_picture(*args, index=0):
    """
    显示一条数据图片
    :param args: train_set_x_orig, train_set_y, classes
    :param index:
    :return:
    """
    # Example of a picture
    train_set_x_orig = args[0]
    train_set_y = args[1]
    classes = args[2]

    train_set_x_item_temp = train_set_x_orig[index]
    class_y = classes[np.squeeze(train_set_y[:, index])].decode("utf-8")
    print(train_set_x_item_temp, class_y)

    plt.imshow(train_set_x_item_temp)
    plt.show()


def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    ### START CODE HERE ### (≈ 1 line of code)
    s = 1 / (1 + np.exp(-z))
    ### END CODE HERE ###

    return s


def sigmoid_test():
    """
    sigmoid function test
    :return:
    """
    print("sigmoid([0, 2]) = " + str(sigmoid(np.array([0, 2]))))


def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    w = np.zeros((dim, 1))
    b = 0
    ### END CODE HERE ###

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


def initialize_with_zeros_test():
    """
    initialize_with_zeros function test
    :return:
    """


def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """

    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    ### START CODE HERE ### (≈ 2 lines of code)
    A = sigmoid(np.dot(w.T, X) + b)
    cost = - np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
    ### END CODE HERE ###

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m
    ### START CODE HERE ### (≈ 2 lines of code)

    ### END CODE HERE ###
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


def start():
    """
    程序入口
    :return:
    """

    # 1、pre-processing

    # Loading the data (cat/non-cat)
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()

    # Example of a picture (matplotlib show)
    # show_example_of_a_picture(train_set_x_orig, train_set_y, classes)

    # data shape
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]
    print("Number of training examples: m_train = " + str(m_train))
    print("Number of testing examples: m_test = " + str(m_test))
    print("Height/Width of each image: num_px = " + str(num_px))
    print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print("train_set_x shape: " + str(train_set_x_orig.shape))
    print("train_set_y shape: " + str(train_set_y.shape))
    print("test_set_x shape: " + str(test_set_x_orig.shape))
    print("test_set_y shape: " + str(test_set_y.shape))
    print()

    # Reshape the training and test examples
    # Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)
    train_set_x_flatten = train_set_x_orig.reshape((train_set_x_orig.shape[0], -1)).T
    test_set_x_flatten = test_set_x_orig.reshape((test_set_x_orig.shape[0], -1)).T

    print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
    print("train_set_y shape: " + str(train_set_y.shape))
    print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
    print("test_set_y shape: " + str(test_set_y.shape))
    print("sanity check after reshaping: " + str(train_set_x_flatten[0:5, 0]))
    print()

    # standardize our dataset
    train_set_x = train_set_x_flatten / 255
    test_set_x = test_set_x_flatten / 255

    # 2、 Building the parts of our algorithm
    # The main steps for building a Neural Network are:
    #
    # Define the model structure (such as number of input features)
    # Initialize the model's parameters
    # Loop:
    #   - Calculate current loss (forward propagation)
    #   - Calculate current gradient (backward propagation)
    #   - Update parameters (gradient descent)


if __name__ == '__main__':
    start()

