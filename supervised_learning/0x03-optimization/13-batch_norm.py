#!/usr/bin/env python3
"""
    Batch Normalization
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Method:
        that normalizes an 'unactivated' output of a neural network
         using batch normalization.

    Args:
        @Z(numpy.ndarray), shape (m, n)
         That should be normalized
        - m: the number of data points
        - n: the number of features in Z

        @gamma(numpy.ndarray), shape (1, n)
         containing the scales used for batch normalization

        @beta:(numpy.ndarray), shape (1, n)
         containing the offsets used for batch normalization

        @epsilon: a small number used to avoid division by zero

    Returns:
        The normalized Z matrix
    """
    mean = np.mean(Z, axis=0)
    variance = np.var((Z - mean), axis=0)
    normalized = (Z - mean) / (np.sqrt(variance + epsilon))
    return gamma * normalized + beta
