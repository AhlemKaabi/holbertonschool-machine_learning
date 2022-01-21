#!/usr/bin/env python3
"""
    Create a Layer with L2 Regularization / tensorFlow
"""
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Method:
        creates a tensorflow layer that includes
          L2 regularization.

    Parameters:
        @prev: a tensor containing the output of the previous layer.
        @n: the number of nodes the new layer should contain.
        @activation: the activation function that should be used on the layer.
        @lambtha: he L2 regularization parameter.

    Returns:
        the output of the new layer.
    """
    init_weights = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer= init_weights,
                            activity_regularizer=tf.keras.regularizers.l2(lambtha),
                            name="layer")
    # regularizer = tf.keras.regularizers.L2(lambtha)
    # output = regularizer(layer(prev))
    output = layer(prev)
    return output