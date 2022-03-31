#!/usr/bin/env python3
"""
    Optimize k
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Method to test for the optimum number of clusters by variance.

    Parameters:
        X (numpy.ndarray of shape(n, d)):
        containing the data set
            n (int): number of data points.
            d (int): number of dimensions for each data point.

        kmin (positive int): containing the minimum number of
        clusters to check for (inclusive).

        kmax (positive int): containing the maximum number of
        clusters to check for (inclusive).

        iterations (positive integer): containing the maximum
        number of iterations for K-means.

    Returns:
        results, d_vars, or None, None on failure

        results (list): containing the outputs of K-means for each
        cluster size
        d_vars (list): containing the difference in variance from
        the smallest cluster size for each cluster size

    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin < 1:
        return None, None
    if not isinstance(kmax, int) or kmax < 1:
        return None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None
    if kmax - kmin < 2:
        return None, None

    d_vars = []
    results = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k)
        # difference in variance from the smallest cluster size
        if k == kmin:
            var_kmin = variance(X, C)
        var = variance(X, C)
        d_vars.append(var_kmin - var)
        results.append((C, clss))
    return results, d_vars
