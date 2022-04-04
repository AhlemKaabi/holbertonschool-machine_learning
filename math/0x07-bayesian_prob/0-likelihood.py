#!/usr/bin/env python3
""" Bayesian Probability - Likelihood

    You are conducting a study on a revolutionary cancer drug
    and are looking to find the probability that a patient who
    takes this drug will develop severe side effects.
    During your trials, n patients take the drug
    and x patients develop severe side effects.


    You can assume that x follows a binomial distribution.

"""
import numpy as np


def likelihood(x, n, P):
    """
    Method:
        Calculates the likelihood of obtaining this data given
        various hypothetical probabilities of developing
        severe side effects.

    Args:
        x: The number of patients that develop severe
        side effects.

        n: The total number of patients observed.

        P: [1D numpy.ndarray] containing the various
        hypothetical probabilities of developing severe
        side effects.

    Raises:
        ValueError:
            - If n is not a positive integer.
            - If x is not an integer that is greater
                than or equal to 0
            - If x is greater than n.
            - If any value in P is not in the range [0, 1]
        TypeError:
            - If P is not a 1D numpy.ndarray.

    Returns:
        1D numpy.ndarray containing the likelihood of
        obtaining the data, x and n, for each probability
        in P, respectively.
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
    for p in P:
        if p < 0 or p > 1:
            raise ValueError("All values in P must be in the range [0, 1]")

	# https://onlinestatbook.com/2/probability/binomial.html
    combinations = np.math.factorial(n) / (
        np.math.factorial(x) * np.math.factorial(n - x)
    )
    P_x = (P ** x) * ((1 - P) ** (n - x))
    return combinations * P_x
