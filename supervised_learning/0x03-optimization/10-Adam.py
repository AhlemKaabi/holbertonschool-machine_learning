#!/usr/bin/env python3
"""
    Adam Upgraded
"""
import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Method:
        That creates the training operation for a neural
         network in tensorflow using the Adam optimization
      algorithm:

    Args:
        @loss: the loss of the network
        @alpha: the learning rate
        @beta1: the weight used for the first moment
        @beta2: the weight used for the second moment
        @epsilon: a small number to avoid division by zero

    Returns:
        The Adam optimization operation
    """
    Adam_train = tf.train.AdamOptimizer(alpha, beta1,
                                        beta2,
                                        epsilon=epsilon)
    Adam_opt_ops = Adam_train.minimize(loss)
    return Adam_opt_ops
