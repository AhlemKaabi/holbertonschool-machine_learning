#!/usr/bin/env python3
"""
    Inception Network
"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Method:
        That builds the inception network as described
         in Going Deeper with Convolutions (2014).

    Args:
        None

     Returns:
         The keras model
    """
    # All convolutions inside and outside the inception block
    # should use a rectified linear activation (ReLU)

    init = K.initializers.HeNormal()
    input_layer = K.Input(shape=(224, 224, 3))

    convolution0_7x7 = K.layers.Conv2D(64, 7, strides=(2, 2),
                                       padding='same',
                                       activation='relu',
                                       kernel_initializer=init)(input_layer)
    # print("conv0 shape", convolution0.shape)
    # conv0 shape (None, 112, 112, 64)

    max_pool0 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                      strides=2,
                                      padding='same')(convolution0_7x7)
    # print("max pooling 0 shape: ", max_pool0.shape)
    # max pooling 0 shape:  (None, 56, 56, 64)

    convol1_1x1 = K.layers.Conv2D(64, 1, strides=1,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer=init)(max_pool0)

    convolution2_3x3 = K.layers.Conv2D(192,
                                       3,
                                       strides=1,
                                       padding='same',
                                       activation='relu',
                                       kernel_initializer=init)(convol1_1x1)
    # print("conv 1 shape", convolution1.shape)
    # # conv 1 shape (None, 56, 56, 192)

    max_pool1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                      strides=2,
                                      padding='same')(convolution2_3x3)
    # print("max pooling 1 shape: ", max_pool1.shape)
    # # max pooling 1 shape:  (None, 28, 28, 192)

    inception_3a = inception_block(max_pool1,
                                   [64, 96, 128, 16, 32, 32])
    # print("-------------------", inception_3a.shape)

    inception_3b = inception_block(inception_3a,
                                   [128, 128, 192, 32, 96, 64])
    # print("------------------------", inception_3b.shape)

    max_pool2 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=2,
                                      padding='same')(inception_3b)
    # print(max_pool2.shape)

    inception_4a = inception_block(max_pool2, [192, 96, 208, 16, 48, 64])
    # print("4a---------------------------", inception_4a.shape)

    inception_4b = inception_block(inception_4a, [160, 112, 224, 24, 64, 64])
    # print("4b----------------------------", inception_4b.shape)

    inception_4c = inception_block(inception_4b, [128, 128, 256, 24, 64, 64])
    # print("4c--------------------", inception_4c.shape)

    inception_4d = inception_block(inception_4c, [112, 144, 288, 32, 64, 64])
    # print("4d-------------------------------", inception_4d.shape)

    inception_4e = inception_block(inception_4d, [256, 160, 320, 32, 128, 128])
    # print("4e---------------", inception_4e.shape)

    max_pool3 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=2,
                                      padding='same')(inception_4e)
    # print(max_pool3.shape)

    inception_5a = inception_block(max_pool3, [256, 160, 320, 32, 128, 128])
    # print("5a----------------------", inception_5a.shape)

    inception_5b = inception_block(inception_5a, [384, 192, 384, 48, 128, 128])
    # print("5b------------------", inception_5b.shape)

    avg_pool0 = K.layers.AveragePooling2D(pool_size=(7, 7),
                                          strides=1)(inception_5b)
    # print(avg_pool0.shape)

    dropout0 = K.layers.Dropout(0.4)(avg_pool0)
    print("dropout shape", dropout0.shape)
    # Flatten = K.layers.Flatten()(dropout0)
    # print(Flatten.shape)

    linear_softmax = K.layers.Dense(1000, activation='softmax')(dropout0)
    # print(linear_softmax.shape)

    model = K.models.Model(input_layer, linear_softmax)

    return model
