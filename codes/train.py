#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/1 20:12
@File:          train.py
'''

import math
import numpy as np
import cv2
from keras.layers import Input, Lambda
from keras import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import Callback

from Loss import Loss
from Dataset import Dataset
from mnist_dataset import get_mnist
from data_generator import data_generator
from discriminator_model import discriminator_model
from generator_model import generator_model
from AddNoise import AddNoise

class WGANGPLoss(Loss):
    def compute_loss(self, inputs):
        lamb = 10
        x, x_score, x_real_score, x_fake_score, x_fake_ng_score, y_pred = inputs
        grad = K.gradients(x_score, [x])[0]
        grad_norm = K.sqrt(K.sum(K.square(grad), axis=[1, 2, 3]))
        grad_pen = K.mean(K.square(grad_norm - 1)) * lamb
        d_loss = K.mean(x_real_score - x_fake_ng_score)
        g_loss = K.mean(x_fake_score - x_fake_ng_score)
        return K.mean(grad_pen + d_loss + g_loss)

if __name__ == '__main__':
    batch_size = 128
    init_lr = 1e-5
    img_size = (28, 28, 1)
    dst_img_size = (140, 140)
    latent_dim = 100

    (X_train, Y_train), _ = get_mnist()
    X_train = X_train[Y_train == 8]
    X_train = X_train.astype('float32') / 127.5 - 1
    X_train = np.expand_dims(X_train, 3)
    dataset = Dataset(X_train)
    generator = data_generator(dataset, batch_size=batch_size, shuffle=True)

    d_input = Input(shape=img_size, dtype='float32')
    d_out = discriminator_model(d_input)
    d_model = Model(d_input, d_out)

    g_input = Input(shape=(latent_dim, ), dtype='float32')
    g_out = generator_model(g_input)
    g_model = Model(g_input, g_out)

    x_in = Input(shape=img_size, dtype='float32')
    z_in = Input(shape=(latent_dim,), dtype='float32')

    x_real = x_in
    x_fake = g_model(z_in)
    x_fake_ng = Lambda(K.stop_gradient)(x_fake)
    x = AddNoise()([x_real, x_fake_ng])

    x_real_score = d_model(x_real)
    x_fake_score = d_model(x_fake)
    x_fake_ng_score = d_model(x_fake_ng)
    x_score = d_model(x)

    out = Lambda(lambda inputs: K.concatenate(inputs))([x_real_score, x_fake_score, x_fake_ng_score])
    out = WGANGPLoss(output_axis=-1)([x, x_score, x_real_score, x_fake_score, x_fake_ng_score, out])

    train_model = Model([x_in, z_in], out)
    opt = Adam(learning_rate=init_lr)
    train_model.compile(opt)

    def evaluate():
        random_latent_vector = np.random.normal(size=(1, latent_dim))
        generated_image = g_model.predict_on_batch(random_latent_vector)[0]

        img = cv2.resize(np.around((generated_image + 1) * 127.5).astype('uint8'), dst_img_size)
        cv2.imwrite('generated_image.png', img)

    class Evaluator(Callback):
        def __init__(self):
            super(Evaluator, self).__init__()

        def on_epoch_end(self, epoch, logs=None):
            evaluate()

    evaluator = Evaluator()

    train_model.fit_generator(
        generator,
        steps_per_epoch=math.ceil(len(X_train) / batch_size),
        epochs=150,
        callbacks=[evaluator],
        shuffle=False,
        initial_epoch=0
    )