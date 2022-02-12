#!/usr/bin/env python3
"""
    RMSProp Upgraded
"""
import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Method:
        That creates the training operation for a neural
         network in tensorflow using the RMSProp optimization algorithm:

    Args:
        loss: the loss of the network
        alpha: the learning rate
        beta2: the RMSProp weight
        epsilon: a small number to avoid division by zero

    Returns:
        The RMSProp optimization operation
    """
    RMSProp = tf.train.RMSPropOptimizer(alpha, beta2,
                                        epsilon=epsilon)

    RMSProp_opt_ops = RMSProp.minimize(loss)
    return RMSProp_opt_ops
