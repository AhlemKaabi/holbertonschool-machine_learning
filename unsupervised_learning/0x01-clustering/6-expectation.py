#!/usr/bin/env python3
"""
    Expectation -xpectation step in the EM algorithm
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Method to calculate the expectation step in the EM algorithm
    for a Gaussian Mixture Model.

    Parameters:
        X (numpy.ndarray of shape (n, d)):
         containing the data set.
        pi (numpy.ndarray of shape (k,)):
         containing the *priors* for each cluster.
        m (numpy.ndarray of shape (k, d)):
         containing the *centroid means* for each cluster.
        S (numpy.ndarray of shape (k, d, d)):
         containing the *covariance* matrices for each cluster.

    Returns:
        g, l, or None, None on failure

        g (numpy.ndarray of shape (k, n)):
        containing the posterior probabilities for each data
        point in each cluster.
        l (float): total log likelihood.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    k, _ = m.shape
    n, d = X.shape

    if k > n:
        return None, None
    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None
    # sum of all pi elem == 1 if not -> retun
    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None

    probs = np.zeros((k, n))
    for i in range(k):
        probs[i] = pi[i] * pdf(X, m[i], S[i])

    sum_probs = np.sum(probs, axis=0)
    g = probs / sum_probs

    log_likelihood = np.sum(np.log(sum_probs))

    return g, log_likelihood
