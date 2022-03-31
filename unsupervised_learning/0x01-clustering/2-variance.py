#!/usr/bin/env python3
"""
    Variance

    ***

    Intra-cluster distance: the distance between members
    of a cluster.

    ***
"""
import numpy as np


def variance(X, C):
    """
    Method to calculate the total intra-cluster variance
    for a data set.

    Parameters:
        X (numpy.ndarray of shape(n, d)):
        containing the data set
            n (int): number of data points.
            d (int): number of dimensions for each data point.

        C (numpy.ndarray of shape (k, d)):
        containing the centroid means for each cluster.

    Returns:
        var (float): the total variance
        or None on failure
    """
    # https://stats.stackexchange.com/questions/86645/variance-within-each-cluster
    # https://numpy.org/doc/stable/reference/generated/numpy.apply_along_axis.html
    # https://numpy.org/doc/stable/reference/generated/numpy.subtract.html
    if (not isinstance(C, np.ndarray)) or (len(X.shape) != 2):
        return None
    if (not isinstance(C, np.ndarray)) or (len(C.shape) != 2):
        return None

    sub = np.apply_along_axis(np.subtract, 1, X, C)
    # X(250x2) C(5x2) => (250x5x2)
    return (np.square(sub).sum(axis=2).min(axis=1).sum())
