#!/usr/bin/env python3
"""
    Policy Gradients
"""

import numpy as np

def policy(matrix, weight):
    """
    ----- Simple Policy function -----
    Method:
    -------
        Computes to policy with a weight of a matrix,
        using softmax.

    Parameters:
    -----------
        matrix: current observation of the environment.
        weight: matrix of random weight.

    Returns:
    --------
        Weighted policy.
    """
    exp = np.exp(matrix.dot(weight))
    # return the probability of the actions 0 and 1.
    return exp / np.sum(exp)