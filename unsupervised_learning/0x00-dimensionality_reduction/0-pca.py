#!/usr/bin/env python3
"""
    PCA -  Principal Component Analysis
"""
import numpy as np


def pca(X, var=0.95):
    """
    Method:
        Performs PCA on a dataset

    Args:
        X[numpy.ndarray], shape (n, d):
        - n: the number of data points
        - d: the number of dimensions in each point
        var: the fraction of the variance that the
          PCA transformation should maintain.

    Returns:
        The weights matrix, W, that maintains var fraction
         of X‘s original variance.

        W[numpy.ndarray], shape(d, nd)
        - nd: new dimensionality of the transformed X
    """
    # We will calculate the W (Eigendecomposition) using the SVD
    # without Computing the (X.transpose X) covariance matrix
    # SVD: Singular Value Decomposition

    _, s, vh = np.linalg.svd(X)

    cumsum = np.cumsum(s)
    sum = np.sum(s)
    variance_fraction = cumsum / sum
    n = 0
    for idx, var_fra in enumerate(variance_fraction):
        if var_fra >= var:
            n = idx
            break

    W = vh.T[:, :n + 1]
    return W
