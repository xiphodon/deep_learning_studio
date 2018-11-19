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


def start():
    """
    程序入口
    :return:
    """
    # Loading the data (cat/non-cat)
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()

    # Example of a picture
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


if __name__ == '__main__':
    start()
