#!/usr/bin/env python3
"""
    DenseNet-121 - Densely Connected Convolutional Networks

"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Method:
        builds the DenseNet-121 architecture.

    Args:
        @growth_rate: the growth rate
        @compression: the compression factor

    Return:
        the keras model
    """
    init = K.initializers.HeNormal()

    input_layer = K.Input(shape=(224, 224, 3))

    batch_norm = K.layers.BatchNormalization()(input_layer)

    activation = K.layers.Activation('relu')(batch_norm)

    convolution = K.layers.Conv2D(64, kernel_size=(7, 7),
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=init)(activation)

    max_pooling = K.layers.MaxPooling2D(pool_size=(3, 3), strides=2,
                                        padding='same')(convolution)

    dense_block_1 = dense_block(max_pooling, 64, growth_rate, 6)

    # print(dense_block_1)
    transition_black_1 = transition_layer(dense_block_1[0],
                                          dense_block_1[1],
                                          compression)

    dense_block_2 = dense_block(transition_black_1[0],
                                transition_black_1[1],
                                growth_rate, 12)

    transition_black_2 = transition_layer(dense_block_2[0],
                                          dense_block_2[1],
                                          compression)

    dense_block_3 = dense_block(transition_black_2[0],
                                transition_black_2[1],
                                growth_rate, 24)

    transition_black_3 = transition_layer(dense_block_3[0],
                                          dense_block_3[1],
                                          compression)

    dense_block_4 = dense_block(transition_black_3[0],
                                transition_black_3[1],
                                growth_rate, 16)

    avg_pooling = K.layers.AveragePooling2D(pool_size=(7, 7),
                                            strides=1)(dense_block_4[0])

    output_layer = K.layers.Dense(1000,
                                  activation='softmax')(avg_pooling)

    model = K.Model(input_layer, output_layer)

    return model
