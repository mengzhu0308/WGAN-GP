#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/23 11:45
@File:          AddNoise.py
'''

from keras import backend as K
from keras.layers import Layer

class AddNoise(Layer):
    def call(self, inputs, **kwargs):
        x_real, x_fake = inputs
        epsilon = K.random_normal((K.shape(x_real)[0], 1, 1, 1), dtype=K.dtype(x_real))
        return x_real * epsilon + x_fake * (1 - epsilon)

    def compute_output_shape(self, input_shape):
        return input_shape[0]