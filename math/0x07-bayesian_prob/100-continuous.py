#!/usr/bin/env python3
""" Bayesian Probability - Continuous Posterior """
from scipy import special


def posterior(x, n, p1, p2):
    """
    Method to calculate the posterior probability that the
    probability of developing severe side effects falls within
    a specific range given the data.

    Parameters:
        x (int): the number of patients that develop severe
          side effects.

        n (int): the total number of patients observed.

        p1 (float): lower bound on the range.

        p2 (float):  upper bound on the range.

	Returns:
		Returns: the posterior probability that p is within the
  		range [p1, p2] given x and n.
    """
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        text = "x must be an integer that is greater than or equal to 0"
        raise ValueError(text)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if (not isinstance(p1, float)) or p1 < 0 or p1 > 1:
        raise ValueError("p1 must be a float in the range [0, 1]")
    if (not isinstance(p2, float)) or p2 < 0 or p2 > 1:
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")
    a = x + 1
    b = n - x + 1
    ac1 = special.btdtr(a, b, p1)
    ac2 = special.btdtr(a, b, p2)
    return ac2 - ac1
