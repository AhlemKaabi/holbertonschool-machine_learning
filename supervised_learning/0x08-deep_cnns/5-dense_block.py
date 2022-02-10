#!/usr/bin/env python3
"""
    Dense Block - Densely Connected Convolutional Networks
"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Method:
        That builds a dense block as described in
         Densely Connected Convolutional Networks.

    Args:
        @X: the output from the previous layer.
        @nb_filters: an integer representing the number of
             filters in X
        @growth_rate: the growth rate for the dense block.
        @layers: the number of layers in the dense block.

    Returns: The concatenated output of each layer within the
        Dense Block and the number of filters within the concatenated
        outputs, respectively
    """
    # the input to the next layer is
    # the concatination of all the previous layer inputs

    # You should use the bottleneck layers used for DenseNet-B
    # All weights should use he normal initialization
    # All convolutions should be preceded by Batch Normalization
    # and a rectified linear activation (ReLU), respectively.
    init = K.initializers.HeNormal()

    output_prev = X
    for _ in range(layers):
        # Basic DenseNet Composition Layer

        # For each composition layer, Pre-Activation Batch Norm (BN) and ReLU,
        # then 3×3 Conv are done with output feature maps of k channels

        batch_norm = K.layers.BatchNormalization()(output_prev)

        activation = K.layers.Activation('relu')(batch_norm)
        # DenseNet-B (Bottleneck Layers)
        # To reduce the model complexity and size,
        # BN-ReLU-1×1 Conv is done before BN-ReLU-3×3 Conv.
        conv2 = K.layers.Conv2D(growth_rate * 4,
                                1,
                                padding='same',
                                kernel_initializer=init)(activation)

        batch_norm_1 = K.layers.BatchNormalization()(conv2)

        activation_1 = K.layers.Activation('relu')(batch_norm_1)

        conv2_1 = K.layers.Conv2D(growth_rate, 3,
                                  padding='same',
                                  kernel_initializer=init)(activation_1)

        concat_output_prev = K.layers.Concatenate()([output_prev, conv2_1])

        output_prev = concat_output_prev

        nb_filters += growth_rate

    return concat_output_prev, nb_filters
