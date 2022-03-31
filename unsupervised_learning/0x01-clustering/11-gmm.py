#!/usr/bin/env python3
"""
    GMM
"""
import sklearn.mixture


def gmm(X, k):
    """
    Method to calculate a GMM from a dataset.

    Parameters:
        X (numpy.ndarray of shape (n, d)):
         containing the dataset
        k (int):
         the number of clusters
    Returns:
        (pi, m, S, clss, bic)
        pi (numpy.ndarray of shape (k,)):
         the cluster priors
        m (numpy.ndarray of shape (k, d)):
           containing the centroid means
        S (numpy.ndarray of shape (k, d, d)):
           containing the covariance matrices
        clss (numpy.ndarray of shape (n,)):
           containing the cluster indices for each data point
        bic (numpy.ndarray of shape (kmax - kmin + 1)):
           containing the BIC value for each cluster size tested
    """
    model = sklearn.mixture.GaussianMixture(n_components=k).fit(X)

    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)

    return pi, m, S, clss, bic
