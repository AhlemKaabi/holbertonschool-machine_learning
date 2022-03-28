#!/usr/bin/env python3
"""
    Initialize K-means

    The cluster centroids should be initialized with
    a multivariate uniform distribution along each dimension in d:

        - The minimum values for the distribution should be
        the minimum values of X along each dimension in d.

        - The maximum values for the distribution should be
        the maximum values of X along each dimension in d

        - You should use numpy.random.uniform exactly once

"""
import numpy as np


def initialize(X, k):
    """
    Method to initializes cluster centroids for K-means.

    Parameters:
        X (numpy.ndarray of shape(n, d)):
        The dataset that will be used for K-means clustering.
            n (int): number of data points.
            d (int): number of dimensions for each data point.

        K (positive int): The number of clusters.

    Returns:
        (numpy.ndarray of shape(k, d)):
        The initialized centroids for each cluster,
        or None on failure.
    """
    _, d = X.shape
    # minimum across rows (axis 0)
    min_ = np.ndarray.min(X, axis=0)
    # maximum across rows (axis 0)
    max_ = np.ndarray.max(X, axis=0)

    output_shape = (k, d)

    centroids = np.random.uniform(min_, max_, output_shape)
    return centroids
