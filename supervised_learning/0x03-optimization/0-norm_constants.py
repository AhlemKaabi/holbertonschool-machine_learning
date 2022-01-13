#!/usr/bin/env python3
"""
    Normalization Constants
"""
import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization (standardization) constants
    of a matrix.

    Parameters:
        X (numpy.ndarray): of shape(m, nx) to normalize
        m (int): number of data points
        nx (int): number of features

    Returns:
        the mean and standard deviation of each feature,
          respectively.
    """
    mean = np.mean(X, axis=0)
    std_deviation = np.std(X, axis=0)

    return mean, std_deviation
