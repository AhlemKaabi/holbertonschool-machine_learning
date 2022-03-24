#!/usr/bin/env python3
""" Bayesian Probability """


import numpy as np


def posterior(x, n, P, Pr):
    """
    update Docstring

    """
    return intersection(x, n, P, Pr) / marginal(x, n, P, Pr)


def marginal(x, n, P, Pr):
    """

    update Docstring

    """
    return np.sum(intersection(x, n, P, Pr))


def intersection(x, n, P, Pr):
    """
    update Docstring

    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")

    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    for pTe in P:
        if pTe < 0 or pTe > 1:
            raise ValueError(
                "All values in P must be in the range [0, 1]"
            )

    for prior in Pr:
        if prior < 0 or prior > 1:
            raise ValueError(
                "All values in Pr must be in the range [0, 1]"
            )

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Compute the likelihood of x given n
    res = np.math.factorial(n) / (
        np.math.factorial(x) * np.math.factorial(n-x)
    )
    likelihood = \
        P ** x * res \
        * (1-P) ** (n-x)

    return likelihood * Pr
