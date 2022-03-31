#!/usr/bin/env python3
"""
    K-means
"""
import numpy as np


def initialize(X, k):
    """
    Method to initialize cluster centroids for K-means.

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
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(k, int) or k <= 0:
        return None
    _, d = X.shape
    # minimum across rows (axis 0)
    min_ = np.ndarray.min(X, axis=0)
    # maximum across rows (axis 0)
    max_ = np.ndarray.max(X, axis=0)

    output_shape = (k, d)

    centroids = np.random.uniform(min_, max_, output_shape)
    return centroids

def kmeans(X, k, iterations=1000):
    centroids = initialize(X, k)

    if centroids is None:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape


    for itr in range(iterations):
        old_centroids = centroids


        distance = np.zeros((n, k))
        for k in range(k):
            row_norm = np.linalg.norm(X - old_centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)

        try:
            labels = np.argmin(distance, axis=1)
        except ValueError:
            pass

        centroids = np.zeros((k, d))
        for k in range(k):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)

        # break if centroids did not change!

        if (np.array_equal(old_centroids, centroids)):
            return centroids, labels

    return centroids, labels
# https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a
