#!/usr/bin/env python3
"""
    Mean and Covariance
"""
import numpy as np


def mean_cov(X):
    """
    Method:
        Calculates the mean and covariance of a data set.

    Args:
        X[numpy.ndarray]:
        - shape (n, d)
        n: number of data points
        d: number of dimension in each data point

    Raises:
        - TypeError: if X is not a 2D numpy.ndarray
        - ValueError: If n is less than 2

    Returns:
        mean, cov:
        - mean[numpy.ndarray], shape (1, d)
        - cov[numpy.ndarray], shape (d, d)
    """

    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a 2D numpy.ndarray")

    if len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    n, d = X.shape
    mean = np.mean(X, axis=0).reshape(1, d)
    term = X - mean
    cov_X_X = np.dot(term.T, term) / (n - 1)
    return mean, cov_X_X