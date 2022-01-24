#!/usr/bin/env python3
"""
    Early Stopping
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Method:
        determines if you should stop gradient descent early.

    Parameters:
        @cost: the current validation cost
        @opt_cost: the lowest recorded validation cost
        @threshold: the threshold used for early stopping
        @patience: the patience count used for early stopping
        @count: the count of how long the threshold has
            not been met

    Returns:
        a boolean of whether the network should be stopped early,
          followed by the updated count
    """
    # Early stopping should occur when:
    # the validation cost of the network
    # has not decreased relative to
    # the optimal validation cost
    # by more than the threshold
    # over a specific patience count
    # opt_cost - cost > threshold and count+1 < patience

    if opt_cost - cost <= threshold and count+1 >= patience:
        return True, count+1
    else:
        return False, count+1

