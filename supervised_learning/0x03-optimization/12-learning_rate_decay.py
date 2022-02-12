#!/usr/bin/env python3
"""
    Learning Rate Decay Upgraded
"""
import tensorflow.compat.v1 as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Method:
        That creates a learning rate decay operation
          in tensorflow using inverse time decay:

    Args:
        @alpha: the original learning rate

        @decay_rate: the weight used to determine the rate at
          which alpha will decay

        @global_step: the number of passes (epochs!)of gradient descent
          that have elapsed

         @decay_step: the number of passes of gradient descent that should
          occur before alpha is decayed further

    Returns:
        The learning rate decay operation
    """
    # the learning rate decay should occur in a stepwise fashion
    # staircase: Whether to apply decay in a discrete staircase,
    # as opposed to continuous, --> fashion.
    return tf.train.inverse_time_decay(learning_rate=alpha,
                                       global_step=global_step,
                                       decay_steps=decay_step,
                                       decay_rate=decay_rate,
                                       staircase=True)
