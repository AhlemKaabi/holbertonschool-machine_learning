#!/usr/bin/env python3
"""
    Initialize GMM - Gaussian Mixture Model
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
        Method to initialize variables for a Gaussian Mixture Mode

        Parameters:
            X (numpy.ndarray of shape(n, d)): The dataset.
                n (int): number of data points.
                d (int): number of dimensions for each data point.

            k (positive int): The number of clusters.

        Returns:
              pi, m, S, or None, None, None on failure

            pi (numpy.ndarray of shape (k,)):
               containing the priors for each cluster, initialized evenly
            m is a (numpy.ndarray of shape (k, d)):
              containing the centroid means for each cluster, initialized
              with K-means
            S (numpy.ndarray of shape (k, d, d)):
              containing the covariance matrices for each
              cluster, initialized as identity matrices
        """
    # https://www.youtube.com/watch?v=JNlEIEwe-Cg
    # https://www.youtube.com/watch?v=q71Niz856KE
    # pi: prior probabilities
    # https://www.vlfeat.org/overview/gmm.html

    # prior: what % of instances came from source c.
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    if not isinstance(k, int) or k <= 0:
        return None, None, None

    _, d = X.shape

    pi = np.ones(k) / k

    m, _ = kmeans(X, k, iterations=1000)

    S = np.tile(np.identity(d), (k, 1)).reshape(k, d, d)

    return pi, m, S
