#!/usr/bin/env python3
"""
    Batch Normalization Upgraded
"""
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Method:
        that creates a batch normalization layer for a
         neural network in tensorflow:

    Args:
        @prev is the activated output of the previous layer
        @n is the number of nodes in the layer to be created
        @activation is the activation function that should be
        used on the output of the layer
        you should use the tf.keras.layers.Dense layer as the base layer with kernal initializer tf.keras.initializers.VarianceScaling(mode='fan_avg')
        your layer should incorporate two trainable parameters, gamma and beta, initialized as vectors of 1 and 0 respectively
        you should use an epsilon of 1e-8

    Returns:
        a tensor of the activated output for the layer
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    layer = tf.keras.layers.Dense(units=n, kernel_initializer = initializer)

    # Calculate the mean and variance of layer
    # for simple batch normalization pass axes=[0] (batch only).
    mean, variance = tf.nn.moments(layer(prev), axes=[0])

    # gamma and beta, initialized as vectors of 1 and 0 respectively
    gamma = tf.ones([n])
    beta = tf.zeros([n])

    epsilon = 1e-8
    batch_normalization_output = tf.nn.batch_normalization(layer(prev), mean,
                                                           variance, beta,
                                                           gamma, epsilon)
    return activation(batch_normalization_output)