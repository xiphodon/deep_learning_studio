#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/12 18:50
# @Author  : GuoChang
# @Site    : https://github.com/xiphodon
# @File    : dl_02_03.py
# @Software: PyCharm

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from dl_02.dl_02_03.tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

np.random.seed(1)


def loss_example_with_tensorflow():
    """
    tensorflow 实现损失函数例子
    :return:
    """
    y_hat = tf.constant(36, name='y_hat')  # Define y_hat constant. Set to 36.
    y = tf.constant(39, name='y')  # Define y. Set to 39

    loss = tf.Variable((y - y_hat) ** 2, name='loss')  # Create a variable for the loss

    init = tf.global_variables_initializer()  # When init is run later (session.run(init)),
    # the loss variable will be initialized and ready to be computed
    with tf.Session() as session:  # Create a session and print the output
        session.run(init)  # Initializes the variables
        print(session.run(loss))  # Prints the loss


def exercise_easy():
    """
    练习(简单)
    :return:
    """
    a = tf.constant(2)
    b = tf.constant(10)
    c = tf.multiply(a, b)
    # print(c)

    sess = tf.Session()
    print(sess.run(c))

    # Change the value of x in the feed_dict

    x = tf.placeholder(tf.int64, name='x')
    print(sess.run(2 * x, feed_dict={x: 3}))
    sess.close()


def linear_function():
    """
    Implements a linear function:
            Initializes W to be a random tensor of shape (4,3)
            Initializes X to be a random tensor of shape (3,1)
            Initializes b to be a random tensor of shape (4,1)
    Returns:
    result -- runs the session for Y = WX + b
    """

    np.random.seed(1)

    ### START CODE HERE ### (4 lines of code)
    X = tf.constant(np.random.randn(3, 1), name='X')
    W = tf.constant(np.random.randn(4, 3), name='W')
    b = tf.constant(np.random.randn(4, 1), name='b')
    Y = tf.add(tf.matmul(W, X), b)
    ### END CODE HERE ###

    # Create the session using tf.Session() and run it with sess.run(...) on the variable you want to calculate

    ### START CODE HERE ###
    sess = tf.Session()
    result = sess.run(Y)
    ### END CODE HERE ###

    # close the session
    sess.close()

    return result


def sigmoid(z):
    """
    Computes the sigmoid of z

    Arguments:
    z -- input value, scalar or vector

    Returns:
    results -- the sigmoid of z
    """

    ### START CODE HERE ### ( approx. 4 lines of code)
    # Create a placeholder for x. Name it 'x'.
    x = tf.placeholder(tf.float32, name='x')

    # compute sigmoid(x)
    sigmoid = tf.sigmoid(x)

    # Create a session, and run it. Please use the method 2 explained above.
    # You should use a feed_dict to pass z's value to x.
    with tf.Session() as sess:
        # Run session and call the output "result"
        result = sess.run(sigmoid, feed_dict={x: z})
    ### END CODE HERE ###

    return result


def cost_tf(logits, labels):
    """
    Computes the cost using the sigmoid cross entropy
    
    Arguments:
    logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
    labels -- vector of labels y (1 or 0)

    Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels"
    in the TensorFlow documentation. So logits will feed into z, and labels into y.
    
    Returns:
    cost -- runs the session of the cost (formula (2))
    """

    ### START CODE HERE ###

    # Create the placeholders for "logits" (z) and "labels" (y) (approx. 2 lines)
    z = tf.placeholder(np.float32, name='z')
    y = tf.placeholder(np.float32, name='y')

    # Use the loss function (approx. 1 line)
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)

    # Create a session (approx. 1 line). See method 1 above.
    sess = tf.Session()

    # Run the session (approx. 1 line).
    cost = sess.run(cost, feed_dict={z: logits, y: labels})

    # Close the session (approx. 1 line). See method 1 above.
    sess.close()

    ### END CODE HERE ###

    return cost


def cost_tf_test():
    """
    cost_tf 测试
    :return:
    """
    # logits = sigmoid(np.array([0.2, 0.4, 0.7, 0.9]))
    logits = np.array([0.2, 0.4, 0.7, 0.9]) # logits 即是 z
    cost = cost_tf(logits, np.array([0, 0, 1, 1]))
    print("cost = " + str(cost))


def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j)
                     will be 1.

    Arguments:
    labels -- vector containing the labels
    C -- number of classes, the depth of the one hot dimension

    Returns:
    one_hot -- one hot matrix
    """

    ### START CODE HERE ###

    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
    C = tf.constant(C, name='C')

    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot_matrix = tf.one_hot(np.squeeze(labels), C, axis=0)

    # Create the session (approx. 1 line)
    sess = tf.Session()

    # Run the session (approx. 1 line)
    one_hot = sess.run(one_hot_matrix)

    # Close the session (approx. 1 line). See method 1 above.
    sess.close()

    ### END CODE HERE ###

    return one_hot


def one_hot_matrix_test():
    """
    one_hot_matrix 测试
    :return:
    """
    labels = np.array([1, 2, 3, 0, 2, 1])
    one_hot = one_hot_matrix(labels, C=4)
    print("one_hot = " + str(one_hot))


def ones(shape):
    """
    Creates an array of ones of dimension shape

    Arguments:
    shape -- shape of the array you want to create

    Returns:
    ones -- array containing only ones
    """

    ### START CODE HERE ###

    # Create "ones" tensor using tf.ones(...). (approx. 1 line)
    ones = tf.ones(shape=shape)

    # Create the session (approx. 1 line)
    sess = tf.Session()

    # Run the session to compute 'ones' (approx. 1 line)
    ones = sess.run(ones)

    # Close the session (approx. 1 line). See method 1 above.
    sess.close()

    ### END CODE HERE ###
    return ones


def ones_test():
    """
    ones 测试
    :return:
    """
    print("ones = " + str(ones([3])))


def sigin_model():
    """
    sigin 模型
    :return:
    """
    # Loading the dataset
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    print(X_train_orig.shape)
    print(Y_train_orig.shape)

    # Example of a picture
    index = 0
    plt.imshow(X_train_orig[index])
    print("y = " + str(np.squeeze(Y_train_orig[:, index])))
    plt.show()

    # Flatten the training and test images
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
    # Normalize image vectors
    X_train = X_train_flatten / 255.
    X_test = X_test_flatten / 255.
    # Convert training and test labels to one hot matrices
    # Y_train = convert_to_one_hot(Y_train_orig, 6)
    Y_train = one_hot_matrix(Y_train_orig, 6)
    # Y_test = convert_to_one_hot(Y_test_orig, 6)
    Y_test = one_hot_matrix(Y_test_orig, 6)

    print("number of training examples = " + str(X_train.shape[1]))
    print("number of test examples = " + str(X_test.shape[1]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))

    return X_train, Y_train, X_test, Y_test, classes


def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)

    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    ### START CODE HERE ### (approx. 2 lines)
    X = tf.placeholder(tf.float32, shape=(n_x, None), name='X')
    Y = tf.placeholder(tf.float32, shape=(n_y, None), name='Y')
    ### END CODE HERE ###

    return X, Y


def create_placeholders_test():
    """
    create_placeholders 测试
    :return:
    """
    X, Y = create_placeholders(12288, 6)
    print("X = " + str(X))
    print("Y = " + str(Y))


def initialize_parameters(weight_scale=0.05):
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """

    tf.set_random_seed(1)  # so that your "random" numbers match ours

    ### START CODE HERE ### (approx. 6 lines of code)
    W1 = tf.get_variable('W1', shape=(25, 12288), dtype=tf.float32,
                         initializer=tf.contrib.layers.xavier_initializer(seed=1),
                         regularizer=tf.contrib.layers.l2_regularizer(scale=weight_scale))
    b1 = tf.get_variable('b1', shape=(25, 1), dtype=tf.float32,
                         initializer=tf.zeros_initializer())
    W2 = tf.get_variable('W2', shape=(12, 25), dtype=tf.float32,
                         initializer=tf.contrib.layers.xavier_initializer(seed=1),
                         regularizer=tf.contrib.layers.l2_regularizer(scale=weight_scale))
    b2 = tf.get_variable('b2', shape=(12, 1), dtype=tf.float32,
                         initializer=tf.zeros_initializer())
    W3 = tf.get_variable('W3', shape=(6, 12), dtype=tf.float32,
                         initializer=tf.contrib.layers.xavier_initializer(seed=1),
                         regularizer=tf.contrib.layers.l2_regularizer(scale=weight_scale))
    b3 = tf.get_variable('b3', shape=(6, 1), dtype=tf.float32,
                         initializer=tf.zeros_initializer())
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


def initialize_parameters_test():
    """
    initialize_parameters 测试
    :return:
    """
    tf.reset_default_graph()
    with tf.Session() as sess:
        parameters = initialize_parameters()
        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))


# GRADED FUNCTION: forward_propagation

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3,Z2) + b3
    ### END CODE HERE ###

    return Z3


def forward_propagation_test():
    """
    forward_propagation 测试
    :return:
    """
    tf.reset_default_graph()

    with tf.Session() as sess:
        X, Y = create_placeholders(12288, 6)
        parameters = initialize_parameters()
        Z3 = forward_propagation(X, parameters)
        print("Z3 = " + str(Z3))


def compute_cost(Z3, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    ### START CODE HERE ### (1 line of code)
    softmax_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
    regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    cost = softmax_loss + regularization_loss
    ### END CODE HERE ###

    return cost


def compute_cost_test():
    """
    compute_cost 测试
    :return:
    """
    tf.reset_default_graph()

    with tf.Session() as sess:
        X, Y = create_placeholders(12288, 6)
        parameters = initialize_parameters()
        Z3 = forward_propagation(X, parameters)
        cost = compute_cost(Z3, Y)
        print("cost = " + str(cost))


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001, weight_scale=0.01,
          num_epochs=1500, minibatch_size=32, print_cost=True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep consistent results
    seed = 3  # to keep consistent results
    (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]  # n_y : output size
    costs = []  # To keep track of the cost

    # Create Placeholders of shape (n_x, n_y)
    ### START CODE HERE ### (1 line)
    X, Y = create_placeholders(n_x, n_y)
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

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    ### START CODE HERE ### (1 line)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    ### END CODE HERE ###

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.  # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                ### END CODE HERE ###

                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost is True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost is True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3, axis=0), tf.argmax(Y, axis=0))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters


def model_test():
    """
    model 测试
    :return:
    """
    X_train, Y_train, X_test, Y_test, classes = sigin_model()
    parameters = model(X_train, Y_train, X_test, Y_test, learning_rate=0.00005, num_epochs=2500, weight_scale=0.05)

    ## learning_rate=0.0001, num_epochs=1500, weight_scale=0.01
    ## Train Accuracy:98.98%
    ## Test Accuracy:80.83%

    ## learning_rate=0.0001, num_epochs=1500, weight_scale=0.02
    ## Train Accuracy:97.5%
    ## Test Accuracy:75.83%

    ## learning_rate=0.0001, num_epochs=1500, weight_scale=0.03
    ## Train Accuracy:95.19%
    ## Test Accuracy:80.83%

    ## learning_rate=0.00001, num_epochs=1500, weight_scale=0.05
    ## Train Accuracy:88.43%
    ## Test Accuracy:82.5%

    ## learning_rate=0.00001, num_epochs=2500, weight_scale=0.05
    ## Train Accuracy:93.06%
    ## Test Accuracy:80.83%

    ## learning_rate=0.00005, num_epochs=2500, weight_scale=0.05
    ## Train Accuracy:96.85%
    ## Test Accuracy:86.67%

    ## learning_rate=0.000001, num_epochs=1500, weight_scale=0.08
    ## Train Accuracy:52.69%
    ## Test Accuracy:50%

    ## learning_rate=0.00001, num_epochs=2000, weight_scale=0.1
    ## Train Accuracy:78.24%
    ## Test Accuracy:76.67%

    ## learning_rate=0.0001, num_epochs=1500, weight_scale=0.08
    ## Train Accuracy:80.46%
    ## Test Accuracy:75%

    ## learning_rate=0.00005, num_epochs=1500, weight_scale=0.08
    ## Train Accuracy:90.56%
    ## Test Accuracy:83.33%

    ## learning_rate=0.00002, num_epochs=2500, weight_scale=0.08
    ## Train Accuracy:87.5%
    ## Test Accuracy:79.17%

    ## learning_rate=0.00008, num_epochs=2500, weight_scale=0.08
    ## Train Accuracy:75.09%
    ## Test Accuracy:64.17%

    ## learning_rate=0.0001, num_epochs=2500, weight_scale=0.08
    ## Train Accuracy:81.39%
    ## Test Accuracy:74.17%

    ## learning_rate=0.00005, num_epochs=2500, weight_scale=0.08
    ## Train Accuracy:91.11%
    ## Test Accuracy:83.33%

    ## learning_rate=0.00005, num_epochs=1500, weight_scale=0.1
    ## Train Accuracy:87.5%
    ## Test Accuracy:79.17%

    ## learning_rate=0.000005, num_epochs=2000, weight_scale=0.1
    ## Train Accuracy:74.81%
    ## Test Accuracy:74.17%

    ## learning_rate=0.00001, num_epochs=2000, weight_scale=0.1
    ## Train Accuracy:78.24%
    ## Test Accuracy:76.67%

    ## learning_rate=0.000025, num_epochs=2000, weight_scale=0.1
    ## Train Accuracy:77.04%
    ## Test Accuracy:71.67%

    ## learning_rate=0.0001, num_epochs=2000, weight_scale=0.1
    ## Train Accuracy:88.33%
    ## Test Accuracy:81.67%

    ## learning_rate=0.00015, num_epochs=2000, weight_scale=0.1
    ## Train Accuracy:66.76%
    ## Test Accuracy:59.17%

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
        my_image = scipy.misc.imresize(image, size=(64, 64)).reshape((1, 64 * 64 * 3)).T
        my_image_prediction = predict(my_image, parameters)

        plt.imshow(image)
        print(img_name)
        print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))


def start():
    """
    程序入口
    :return:
    """
    # loss_example_with_tensorflow()
    # exercise_easy()
    # print(linear_function())
    # print(sigmoid(0))
    # print(sigmoid(12))
    # cost_tf_test()
    # one_hot_matrix_test()
    # ones_test()
    # sigin_model()
    # create_placeholders_test()
    # initialize_parameters_test()
    # forward_propagation_test()
    # compute_cost_test()
    model_test()


if __name__ == '__main__':
    start()
