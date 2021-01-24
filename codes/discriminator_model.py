#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/10 11:19
@File:          discriminator_model.py
'''

from keras.layers import *

def discriminator_model(x):
    x = Conv2D(128, 4, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, 4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    x = Dense(1)(x)

    return x