#!/usr/bin/env python3
"""
    Transition Layer - Densely Connected Convolutional Networks

"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Method:
        That builds a transition layer as described in Densely
         Connected Convolutional Networks.

    Args:
        @X: output from the previous layer
        @nb_filters: an integer representing the number
         of filters in X
        @compression: the compression factor for the transition layer

    Returns:
        The output of the transition layer and the number of
         filters within the output, respectively
    """
    # Your code should implement compression as used in DenseNet-C
    # All weights should use he normal initialization
    # All convolutions should be preceded by Batch Normalization
    # and a rectified linear activation (ReLU), respectively
    init = K.initializers.HeNormal()

    output_prev = X

    batch_norm = K.layers.BatchNormalization()(output_prev)

    activation = K.layers.Activation('relu')(batch_norm)

    n_filt = int(nb_filters * compression)
    conv2 = K.layers.Conv2D(n_filt, 1,
                            padding='same',
                            kernel_initializer=init)(activation)

    avg_pooling = K.layers.AveragePooling2D(pool_size=(2, 2), strides=2)(conv2)

    return avg_pooling, n_filt
