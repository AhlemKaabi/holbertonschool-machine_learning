#!/usr/bin/env python3
"""
    Building Inception Architecture Block
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Method:
        That builds an inception block as described in
         Going Deeper with Convolutions (2014)

    Args:
        @A_prev: the output from the previous layer
        @filters: a tuple or list containing:
            - F1: is the number of filters in the 1x1 convolution
            - F3R: is the number of filters in the 1x1 convolution
                before the 3x3 convolution
            - F3: is the number of filters in the 3x3 convolution
            - F5R: is the number of filters in the 1x1 convolution
                before the 5x5 convolution
            - F5: is the number of filters in the 5x5 convolution
            - FPP: is the number of filters in the 1x1 convolution
               after the max pooling

    Returns:
        The concatenated output of the inception block
    """
    # All convolutions inside the inception block should
    # use a rectified linear activation (ReLU)

    F1, F3R, F3, F5R, F5, FPP = filters

    input_layer = A_prev

    init = K.initializers.HeNormal()

    # 1x1xF1 convolution
    F1_layer = K.layers.Conv2D(F1, 1, activation='relu', padding='same',
                               kernel_initializer=init)(input_layer)
    # print("1------------", F1_layer.shape)

    # 1x1xF3R convolution
    F3R_layer = K.layers.Conv2D(F3R, 1, activation='relu', padding='same',
                                kernel_initializer=init)(input_layer)

    # 3x3xF3 convolution
    F3_layer = K.layers.Conv2D(F3, 3, activation='relu', padding='same',
                               kernel_initializer=init)(F3R_layer)

    # 1x1xF5R convolution
    F5R_layer = K.layers.Conv2D(F5R, 1, activation='relu', padding='same',
                                kernel_initializer=init)(input_layer)
    # print("1------------", F1_layer.shape)

    # 3x3xF3 convolution
    F5_layer = K.layers.Conv2D(F5, 5, activation='relu', padding='same',
                               kernel_initializer=init)(F5R_layer)

    # max pooling convolution
    FP_layer = K.layers.MaxPooling2D(pool_size=(3, 3), strides=1,
                                     padding='same')(input_layer)

    # print("fp layer shape: ", FP_layer)
    # 1x1xFPP convolution
    FPP_layer = K.layers.Conv2D(FPP, 1, activation='relu', padding='same',
                                kernel_initializer=init)(FP_layer)

    concatted_layer = K.layers.Concatenate()([F1_layer, F3_layer,
                                              F5_layer, FPP_layer])

    return concatted_layer
