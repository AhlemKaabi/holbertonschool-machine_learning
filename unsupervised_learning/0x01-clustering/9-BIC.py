#!/usr/bin/env python3
"""
    BIC - Bayesian Information Criterion
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000,
        tol=1e-5, verbose=False):
    """
    Method to find the best number of clusters for a GMM
    using the Bayesian Information Criterion.

    Parameters:
        X (numpy.ndarray of shape (n, d))
         containing the data set.

        kmin  (positive integer)
         containing the minimum number of clusters to check for (inclusive).

        kmax  (positive integer)
         containing the maximum number of clusters to check for (inclusive).

        iterations (positive integer)
         containing the maximum number of iterations for the EM algorithm.

        tol (non-negative float)
         containing the tolerance for the EM algorithm.

        verbose  (boolean)
         that determines if the EM algorithm should print information to
         the standard output.

    Returns:

    (best_k, best_result, l, b), or (None, None, None, None) on failure.

    best_k (int): the best value for k based on its BIC.

    best_result  (tuple): containing pi, m, S
        - pi  (numpy.ndarray of shape (k,))
         containing the cluster priors for the best number of clusters.

        - m  (numpy.ndarray of shape (k, d))
         containing the centroid means for the best number of clusters.

        - S  (numpy.ndarray of shape (k, d, d))
         containing the covariance matrices for the best number of clusters.

    l (numpy.ndarray of shape (kmax - kmin + 1))
     containing the log likelihood for each cluster size tested.

    b (numpy.ndarray of shape (kmax - kmin + 1))
     containing the BIC value for each cluster size tested.
        Use: BIC = p * ln(n) - 2 * l
        - p (int) the number of parameters required for the model.
        - n (int) the number of data points used to create the model.
        - l (float) the log likelihood of the model.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None

    if not isinstance(kmin, int) or kmin <= 0 or X.shape[0] <= kmin:
        return None, None, None, None

    if not isinstance(kmax, int) or kmax <= 0 or X.shape[0] <= kmax:
        return None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape

    priors = []
    centroid_means = []
    covariances = []
    log_likelihood = []
    BIC_values = []

    for k in range(kmin, kmax + 1):
        pi, m, S, _, log_lik = expectation_maximization(
            X,
            k,
            iterations,
            tol,
            verbose
        )
        priors.append(pi)
        centroid_means.append(m)
        covariances.append(S)
        log_likelihood.append(log_lik)

        p = (k * d) + k * (d * (d + 1) / 2) + (d * k) + k - 1
        # https://stats.stackexchange.com/questions/436181/number-of-parameters-to-be-learned-in-k-guassian-mixture-model
        BIC_values.append(p * np.log(n) - 2 * log_lik)

    best_k = np.argmin(BIC_values)
    # https://stackoverflow.com/questions/51144580/should-bic-bayesian-information-criterion-be-lower-or-higher
    # https://en.wikipedia.org/wiki/Bayesian_information_criterion
    best_reslt = (priors[best_k], centroid_means[best_k], covariances[best_k])

    return best_k, best_reslt, log_likelihood, BIC_values
