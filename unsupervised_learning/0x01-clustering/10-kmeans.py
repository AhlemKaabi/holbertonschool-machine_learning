#!/usr/bin/env python3
"""
    Hello, sklearn!
"""
import sklearn.cluster


def kmeans(X, k):
    """
    Method to perform K-means on a dataset.

    Parameters:
        X (numpy.ndarray of shape (n, d)) containing the dataset
        k (int)the number of clusters
    Returns: C, clss
        C is a numpy.ndarray of shape (k, d) containing the centroid means for
         each cluster.
        clss is a numpy.ndarray of shape (n,) containing the index of the
         cluster in C that each data point belongs to.
    """
    k_means = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = k_means.cluster_centers_
    clss = k_means.labels_
    return C, clss
