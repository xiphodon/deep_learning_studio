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
    logits = np.array([0.2, 0.4, 0.7, 0.9]) # logits 即是 z
    cost = cost_tf(logits, np.array([0, 0, 1, 1]))
    print("cost = " + str(cost))


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
    cost_tf_test()


if __name__ == '__main__':
    start()
