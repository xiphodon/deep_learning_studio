#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/25 17:54
# @Author  : GuoChang
# @Site    : https://github.com/xiphodon
# @File    : dl_04_04.py
# @Software: PyCharm

# Face Recognition

from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from dl_04.dl_04_04.fr_utils import *
from dl_04.dl_04_04.inception_blocks import *

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 使用CPU

np.set_printoptions(threshold=np.nan)


def start():
    """
    入口
    :return:
    """
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    print("Total Params:", FRmodel.count_params())


if __name__ == '__main__':
    start()
