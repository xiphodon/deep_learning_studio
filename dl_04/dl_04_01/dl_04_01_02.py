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


def initialize_parameters(weight_scale=0.0001):
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """

    tf.set_random_seed(1)  # so that your "random" numbers match ours

    ### START CODE HERE ### (approx. 2 lines of code)
    W1 = tf.get_variable(name='W1', shape=(4, 4, 3, 8),
                         initializer=tf.contrib.layers.xavier_initializer(seed=0),
                         regularizer=tf.contrib.layers.l2_regularizer(scale=weight_scale)) * np.sqrt(2.0)
    W2 = tf.get_variable(name='W2', shape=(2, 2, 8, 16),
                         initializer=tf.contrib.layers.xavier_initializer(seed=0),
                         regularizer=tf.contrib.layers.l2_regularizer(scale=weight_scale)) * np.sqrt(2.0/(4 * 4 * 3 * 8))
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


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']

    ### START CODE HERE ###
    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    # FLATTEN
    F = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    Z3 = tf.contrib.layers.fully_connected(F, num_outputs=6, activation_fn=None)
    ### END CODE HERE ###

    print(Z1)
    print(A1)
    print(P1)
    print(Z2)
    print(A2)
    print(P2)
    print(F)

    return Z3


def forward_propagation_test():
    """
    forward_propagation 测试
    :return:
    """
    tf.reset_default_graph()
    np.random.seed(1)

    with tf.Session() as sess:
        X, Y = create_placeholders(64, 64, 3, 6)
        parameters = initialize_parameters()
        Z3 = forward_propagation(X, parameters)
        init = tf.global_variables_initializer()
        sess.run(init)
        a = sess.run(Z3, {X: np.random.randn(2, 64, 64, 3), Y: np.random.randn(2, 6)})
        print("Z3 = " + str(a))


def compute_cost(Z3, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3, labels=Y))
    regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    cost = cost + regularization_loss
    ### END CODE HERE ###

    return cost


def compute_cost_test():
    """
    compute_cost 测试
    :return:
    """
    tf.reset_default_graph()

    with tf.Session() as sess:
        np.random.seed(1)
        X, Y = create_placeholders(64, 64, 3, 6)
        parameters = initialize_parameters()
        Z3 = forward_propagation(X, parameters)
        cost = compute_cost(Z3, Y)
        init = tf.global_variables_initializer()
        sess.run(init)
        a = sess.run(cost, {X: np.random.randn(4, 64, 64, 3), Y: np.random.randn(4, 6)})
        print("cost = " + str(a))


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.009,
          num_epochs=100, minibatch_size=64, weight_scale=0.0001, print_cost=True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep results consistent (tensorflow seed)
    seed = 3  # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []  # To keep track of the cost

    # Create Placeholders of the correct shape
    ### START CODE HERE ### (1 line)
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters(weight_scale=weight_scale)
    ### END CODE HERE ###

    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z3 = forward_propagation(X, parameters)
    ### END CODE HERE ###

    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(Z3, Y)
    ### END CODE HERE ###

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    ### END CODE HERE ###

    # Initialize all the variables globally
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                ### END CODE HERE ###

                minibatch_cost += temp_cost / num_minibatches

            # Print the cost every epoch
            if print_cost is True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost is True and epoch % 1 == 0:
                costs.append(minibatch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        parameters = sess.run(parameters)

        return train_accuracy, test_accuracy, parameters


def start():
    """
    入口
    :return:
    """
    # Loading the data (signs)
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    # Example of a picture
    # index = 6
    # plt.imshow(X_train_orig[index])
    # print("y = " + str(np.squeeze(Y_train_orig[:, index])))
    # plt.show()

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

    _, _, parameters = model(X_train, Y_train, X_test, Y_test,
                             learning_rate=0.009, num_epochs=300, minibatch_size=64, weight_scale=0.002)

    # learning_rate=0.009, num_epochs=100, minibatch_size=64
    # Train Accuracy: 0.92777777
    # Test Accuracy: 0.80833334

    # learning_rate=0.009, num_epochs=300, minibatch_size=64
    # Train Accuracy: 1.0
    # Test Accuracy: 0.825

    # learning_rate=0.009, num_epochs=300, minibatch_size=64, weight_scale=0.0001
    # Train Accuracy: 1.0
    # Test Accuracy: 0.85833335

    # learning_rate=0.009, num_epochs=300, minibatch_size=64, weight_scale=0.001
    # Train Accuracy: 1.0
    # Test Accuracy: 0.9

    # learning_rate=0.009, num_epochs=300, minibatch_size=64, weight_scale=0.002
    # Train Accuracy: 0.98333335
    # Test Accuracy: 0.89166665

    # learning_rate=0.009, num_epochs=300, minibatch_size=64, weight_scale=0.003
    # Train Accuracy: 0.9768519
    # Test Accuracy: 0.85

    # learning_rate=0.009, num_epochs=300, minibatch_size=64, weight_scale=0.005
    # Train Accuracy: 0.9425926
    # Test Accuracy: 0.8666667

    # learning_rate=0.009, num_epochs=300, minibatch_size=64, weight_scale=0.006
    # Train Accuracy: 0.90925926
    # Test Accuracy: 0.85

    # learning_rate=0.01, num_epochs=300, minibatch_size=64, weight_scale=0.006
    # Train Accuracy: 0.90092593
    # Test Accuracy: 0.725

    # learning_rate=0.005, num_epochs=300, minibatch_size=64, weight_scale=0.01
    # Train Accuracy: 0.8851852
    # Test Accuracy: 0.875

    # learning_rate=0.005, num_epochs=300, minibatch_size=64, weight_scale=0.008
    # Train Accuracy: 0.9064815
    # Test Accuracy: 0.80833334

    # learning_rate=0.005, num_epochs=500, minibatch_size=64, weight_scale=0.01
    # Train Accuracy: 0.84166664
    # Test Accuracy: 0.76666665

    # learning_rate=0.005, num_epochs=300, minibatch_size=64, weight_scale=0.006
    # Train Accuracy: 0.93425924
    # Test Accuracy: 0.8333333

    # learning_rate=0.001, num_epochs=300, minibatch_size=64, weight_scale=0.01
    # Train Accuracy: 0.83148146
    # Test Accuracy: 0.7916667

    # learning_rate=0.001, num_epochs=600, minibatch_size=64, weight_scale=0.01
    # Train Accuracy: 0.88611114
    # Test Accuracy: 0.825

    # learning_rate=0.002, num_epochs=500, minibatch_size=64, weight_scale=0.01
    # Train Accuracy: 0.90833336
    # Test Accuracy: 0.825

    # learning_rate=0.003, num_epochs=400, minibatch_size=64, weight_scale=0.01
    # Train Accuracy: 0.93703705
    # Test Accuracy: 0.85

    # learning_rate=0.004, num_epochs=1000, minibatch_size=64, weight_scale=0.01
    # Train Accuracy: 0.90925926
    # Test Accuracy: 0.8

    # learning_rate=0.003, num_epochs=1000, minibatch_size=64, weight_scale=0.01
    # Train Accuracy: 0.962963
    # Test Accuracy: 0.89166665

    # learning_rate=0.003, num_epochs=1000, minibatch_size=64, weight_scale=0.012
    # Train Accuracy: 0.9398148
    # Test Accuracy: 0.825

    # learning_rate=0.005, num_epochs=1000, minibatch_size=64, weight_scale=0.01
    # Train Accuracy: 0.9398148
    # Test Accuracy: 0.825

    predict_my_image(parameters)


def predict_my_image(parameters):
    """
    预测我的图片
    :return:
    """
    import scipy
    from PIL import Image
    from scipy import ndimage
    import os

    images_dir_path = r'./images/'

    for img_name in os.listdir(images_dir_path):
        img_path = os.path.join(images_dir_path, img_name)

        ## START CODE HERE ## (PUT YOUR IMAGE NAME)
        # my_image = "thumbs_up.jpg"
        ## END CODE HERE ##

        # We preprocess your image to fit your algorithm.
        fname = img_path
        image = np.array(ndimage.imread(fname, flatten=False))
        my_image = scipy.misc.imresize(image, size=(64, 64))[np.newaxis, :]
        print(my_image.shape)
        my_image_prediction, z = model_predict(my_image, parameters)

        plt.imshow(image)
        print(img_name)
        print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))
        print(z)
        plt.show()


def model_predict(X, parameters):
    W1 = tf.convert_to_tensor(parameters["W1"])
    W2 = tf.convert_to_tensor(parameters["W2"])

    params = {"W1": W1,
              "W2": W2}

    x = tf.placeholder(dtype=tf.float32, shape=(None, 64, 64, 3))

    z3 = forward_propagation(x, params)
    p = tf.argmax(z3, axis=1)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        prediction, z = sess.run((p, z3), feed_dict={x: X})

    return prediction, z


if __name__ == '__main__':
    start()
    # create_placeholders_test()
    # initialize_parameters_test()
    # forward_propagation_test()
    # compute_cost_test()
