#!/usr/bin/env python3
"""
    Learning Rate Decay
"""


def learning_rate_decay(alpha, decay_rate,
                        global_step, decay_step):
    """
    Method:
        That updates the learning rate using inverse
         time decay in numpy.
    Args:
        @alpha: the original learning rate

        @decay_rate: the weight used to determine the rate at
          which alpha will decay

        @global_step: the number of passes (epochs!)of gradient descent
          that have elapsed

         @decay_step: the number of passes of gradient descent that should
          occur before alpha is decayed further

    Returns:
        The updated value for alpha
    """
    # the learning rate decay should occur in a stepwise fashion
    # Floor division
    epoch_num = global_step // decay_step
    new_alpha = alpha / (1 + (decay_rate * epoch_num))
    return new_alpha
