#!/usr/bin/env python3
"""
    PCA -  Principal Component Analysis
"""
import numpy as np


def pca(X, ndim):
    """
    Method:
        Performs PCA on a dataset

    Args:
        X[numpy.ndarray], shape (n, d):
        - n: the number of data points
        - d: the number of dimensions in each point
        ndim: The new dimensionality of the transformed X.

    Returns:
        T[numpy.ndarray], shape (n, ndim)
        containing the transformed version of X
    """
    # We will calculate the W (Eigendecomposition) using the SVD
    # without Computing the (X.transpose X) covariance matrix
    # SVD: Singular Value Decomposition
    M = X - np.mean(X, axis=0)
    _, _, vh = np.linalg.svd(X)

    W = vh.T[:, :ndim]
    return np.dot(M, W)
