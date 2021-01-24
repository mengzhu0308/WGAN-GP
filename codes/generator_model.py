#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/10 10:51
@File:          generator_model.py
'''

from keras.layers import *

def generator_model(x):
    x = Dense(7 * 7 * 256, use_bias=False)(x)
    x = Reshape((7, 7, 256))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(1, 4, strides=2, padding='same')(x)
    x = Activation('tanh')(x)

    return x
