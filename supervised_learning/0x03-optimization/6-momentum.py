#!/usr/bin/env python3
"""
    Momentum Upgraded
"""
import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Method:
        That creates the training operation for a neural
         network in tensorflow using the gradient descent with
          momentum optimization algorithm.

    Args:
        @loss: the loss of the network
        @alpha: the learning rate
        beta1: the momentum weight

    Returns:
        The momentum optimization operation
    """
    momentum_train = tf.train.MomentumOptimizer(learning_rate=alpha,
                                                momentum=beta1)
    grad = momentum_train.minimize(loss)
    return grad
