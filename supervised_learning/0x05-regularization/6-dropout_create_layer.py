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
    # class Dropout: Applies Dropout to the input!(layer, weights..)
    # kernel_regularizer: Regularizer to apply a penalty on the layer's kernel
    #   -> Tries to reduce the weights W (excluding bias)
    init_weights = tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg')

    dropout = tf.layers.Dropout(keep_prob)
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=init_weights,)
    # output = layer(prev)
    # applying the dropout not only on the weights but on the whole layer
    output = dropout(layer(prev))
    return output
