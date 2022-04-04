#!/usr/bin/env python3
""" Bayesian Probability - Posterior """
import numpy as np


def intersection(x, n, P, Pr):
    """
    Method to calculate the intersection of obtaining this
    data with the various hypothetical probabilities.

    Parameters:
        x (int): the number of patients that develop severe
          side effects.

        n (int): the total number of patients observed.

        P (1D numpy.ndarray): containing the various hypothetical
          probabilities of developing severe side effects.

        Pr (1D numpy.ndarray) containing the prior beliefs of P.

    Returns:
        (1D numpy.ndarray) containing the intersection of
          obtaining x and n with each probability in P, respectively
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

    for p in P:
        if p < 0 or p > 1:
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

    res = np.math.factorial(n) / (
        np.math.factorial(x) * np.math.factorial(n-x)
    )
    likelihood = \
        P ** x * res \
        * (1-P) ** (n-x)

    return likelihood * Pr


def marginal(x, n, P, Pr):
    """
    Method to calculate the marginal probability of
    obtaining the data.

    Returns:
        the marginal probability of obtaining x and n
    """
    # https://youtu.be/r27mouuyFQk?list=PLFDbGp5YzjqXQ4oE4w9GVWdiokWB9gEpm
    return np.sum(intersection(x, n, P, Pr))


def posterior(x, n, P, Pr):
    """
    Method to calculate the posterior probability for the various
    hypothetical probabilities of developing severe side effects
    given the data.

	Returns:
 		the posterior probability of each probability
 		in P given x and n, respectively.
    """
    return intersection(x, n, P, Pr) / marginal(x, n, P, Pr)
