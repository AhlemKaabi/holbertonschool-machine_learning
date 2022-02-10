#!/usr/bin/env python3
"""
    Identity Block - Deep Residual Learning for Image Recognition
"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Method:
        That builds an identity block as described
         in Deep Residual Learning for Image Recognition (2015).

    Args:
        @A_prev is the output from the previous layer
        @filters is a tuple or list containing.
            - F11: the number of filters in the first 1x1 convolution
            - F3: the number of filters in the 3x3 convolution
            - F12: the number of filters in the second 1x1 convolution
    Returns:
        The activated output of the identity block
    """
    # All convolutions inside the block should be followed by
    # batch normalization along the channels axis and
    # a rectified linear activation (ReLU), respectively.

    # All weights should use he normal initialization
    F11, F3, F12 = filters

    init = K.initializers.HeNormal()

    input_layer = A_prev

    conv2 = K.layers.Conv2D(F11, 1,
                            padding='same',
                            kernel_initializer=init)(input_layer)

    batch_norm = K.layers.BatchNormalization()(conv2)

    activation = K.layers.Activation('relu')(batch_norm)

    conv2_1 = K.layers.Conv2D(F3, 3,
                              padding='same',
                              kernel_initializer=init)(activation)

    batch_norm_1 = K.layers.BatchNormalization()(conv2_1)

    activation_1 = K.layers.Activation('relu')(batch_norm_1)

    conv2_2 = K.layers.Conv2D(F12, 1,
                              padding='same',
                              kernel_initializer=init)(activation_1)

    batch_norm_1 = K.layers.BatchNormalization()(conv2_2)

    add_layer = K.layers.Add()([batch_norm_1, input_layer])

    activation_2 = K.layers.Activation('relu')(add_layer)

    return activation_2
