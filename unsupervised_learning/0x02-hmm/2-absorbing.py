#!/usr/bin/env python3
"""
    Hidden Markov Models - Absorbing Chains
"""
import numpy as np


def absorbing(P):
    """
    Method to determine if a markov chain is absorbing.

    Parameters:
        P ((square 2D numpy.ndarray) of shape (n, n)):
         representing the transition matrix.
        - P[i, j]: the probability of transitioning from state i to state j
        - n : the number of states in the markov chain

    Returns:
        True if it is absorbing, or False on failure
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return False
    n, m = np.shape(P)

    if n != m:
        return False

    if not np.all(np.isclose(P.sum(axis=1), 1)):
        return False

    return np.any(P.diagonal() == 1)
