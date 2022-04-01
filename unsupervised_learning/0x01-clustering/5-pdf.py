#!/usr/bin/env python3
"""
    PDF
"""
import numpy as np


def pdf(X, m, S):
    """
    Method to calculate the probability density function of
    a Gaussian distribution.

    Parameters:
        X (numpy.ndarray of shape (n, d)):
          containing the data points whose PDF should be evaluated
        m (numpy.ndarray of shape (d,)):
          containing the mean of the distribution
        S (numpy.ndarray of shape (d, d)):
          containing the covariance of the distribution

    Returns: P, or None on failure
        P (numpy.ndarray of shape (n,))
        containing the PDF values for each data point.

    ** All values in P should have a minimum value of 1e-300 **
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None
    d = S.shape[0]

    det = np.linalg.det(S)
    inv = np.linalg.inv(S)

    term1 = 1 / np.sqrt(((2 * np.pi) ** d) * det)
    mul = np.dot((X - m), inv)
    exp_term = np.sum(mul * (X - m) / -2, axis=1)
    P = term1 * np.exp(exp_term)
    return np.where(P < 1e-300, 1e-300, P)
