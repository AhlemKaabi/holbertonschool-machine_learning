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
    # https://stats.stackexchange.com/questions/383310/what-is-the-difference-between-kernel-bias-and-activity-regulizers-and-when-t

    # kernel_regularizer: Regularizer to apply a penalty on the layer's kernel
    #   -> Tries to reduce the weights W (excluding bias)
    # bias_regularizer: Regularizer to apply a penalty on the layer's bias
    # activity_regularizer: Regularizer to apply a penalty on the layer's
    #   output -> Tries to reduce the layer's output y, thus will reduce the
    #   weights and adjust bias so Wx+b is smallest.

    # https://stackoverflow.com/questions/51683495/what-does-it-mean-to-set-kernel-regularizer-to-be-l2-regularizer-in-tf-layers-co

    init_weights = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                         mode=("fan_avg"))
    
    regularizer = tf.keras.regularizers.L2(lambtha)
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=init_weights,
                            kernel_regularizer=regularizer,
                            name="layer")
    # output = regularizer(layer(prev))
    output = layer(prev)
    return output
