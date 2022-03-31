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
    """
    Method to perform K-means on a dataset.

    Parameters:
        X (numpy.ndarray of shape(n, d)): The dataset.
            n (int): number of data points.
            d (int): number of dimensions for each data point.

        K (positive int): The number of clusters.

        iterations (positive int): the maximum number of iterations
          that should be performed

    Returns:
        C, clss or None, None on failure

        C (numpy.ndarray of shape(k, d)):
        containing the centroid means for each cluster.

        clss (numpy.ndarray of shape (n,)):
        containing the index of the cluster in C
        that each data point belongs to
    """
    centroids = initialize(X, k)

    if centroids is None:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    _, d = X.shape

    for _ in range(iterations):
        old_centroids = np.copy(centroids)

        distances = np.sqrt(np.sum((X - centroids[:, np.newaxis])**2,
                                   axis=2))

        labels = np.argmin(distances, axis=0)
        # print(distances)
        # print(labels)
        # return the index of each minimum distance
        for j in range(k):
            if (X[labels == j].size == 0):
                centroids[j] = np.random.uniform(low=np.min(X, axis=0),
                                                 high=np.max(X, axis=0),
                                                 size=(1, d))
            else:
                centroids[j] = np.mean(X[labels == j], axis=0)

        if np.all(old_centroids == centroids):
            break

    return centroids, labels
