#!/usr/bin/env python3
"""
    Create a Layer with Dropout 
"""
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Method:
        that creates a layer of a neural network using dropout.

    Parameters:
        @prev: tensor containing the output of the previous layer
        @n: is the number of nodes the new layer should contain
        @activation: is the activation function that should be
            used on the layer
        @keep_prob: is the probability that a node will be kept

    Returns:
        the output of the new layer

    """
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout
    # class Dropout: Applies Dropout to the input.

    init_weights = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    dropout = tf.keras.layers.Dropout(keep_prob)
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=init_weights,
                            kernel_regularizer=dropout,
                            name="layer")
    # output = regularizer(layer(prev))
    output = layer(prev)
    return output
