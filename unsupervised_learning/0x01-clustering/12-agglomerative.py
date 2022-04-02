#!/usr/bin/env python3
"""
    Agglomerative
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Method to perform agglomerative clustering on a dataset.

    Parameters:
        X (numpy.ndarray of shape (n, d))
         containing the dataset.
        dist (int): the maximum cophenetic distance
         for all clusters.

    Returns:
        clss (numpy.ndarray of shape (n,)):
         containing the cluster indices for each data point.

    ** Performs agglomerative clustering with Ward linkage **
    """
    # https://www.kaggle.com/code/vipulgandhi/hierarchical-clustering-explanation/notebook
    # https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage

    Z = scipy.cluster.hierarchy.linkage(X, 'ward')

    dn = scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)

    # distance :
    #   Forms flat clusters so that the original observations
    #   in each flat cluster have no greater a
    #   cophenetic distance than t.

    clss = scipy.cluster.hierarchy.fcluster(Z, t=dist, criterion='distance')

    plt.show()

    return clss
