#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/28 18:28
# @Author  : GuoChang
# @Site    : https://github.com/xiphodon
# @File    : dl_04_01.py
# @Software: PyCharm


import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)


def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image,
    as illustrated in Figure 1.

    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """

    ### START CODE HERE ### (≈ 1 line)
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
    ### END CODE HERE ###

    return X_pad


def zero_pad_test():
    """
    zero_pad 测试
    :return:
    """
    np.random.seed(1)
    x = np.random.randn(4, 3, 3, 2)
    x_pad = zero_pad(x, 2)
    print("x.shape =", x.shape)
    print("x_pad.shape =", x_pad.shape)
    print("x[1,1] =", x[1, 1])
    print("x_pad[1,1] =", x_pad[1, 1])

    fig, axarr = plt.subplots(1, 2)
    axarr[0].set_title('x')
    axarr[0].imshow(x[0, :, :, 0])
    axarr[1].set_title('x_pad')
    axarr[1].imshow(x_pad[0, :, :, 0])
    plt.show()


if __name__ == '__main__':
    zero_pad_test()