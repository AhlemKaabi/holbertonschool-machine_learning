#!/usr/bin/env python3
"""
    Hidden Markov Models - Regular Chains
"""
import numpy as np


def regular(P):
    """
    Method to determine the steady state probabilities of a regular markov
    chain.

    Parameters:
        P ((square 2D numpy.ndarray) of shape (n, n)):
         representing the transition matrix.
        - P[i, j]: the probability of transitioning from state i to state j
        - n : the number of states in the markov chain

    Returns:
        (numpy.ndarray of shape (1, n)): containing the steady state
        probabilities, or None on failure
    """
    # https://www.maplesoft.com/support/help/maple/view.aspx?path=examples%2FSteadyStateMarkovChain
    if type(P) != np.ndarray:
        return None
    if len(P.shape) != 2:
        return None
    try:
        power = np.linalg.matrix_power(P, 50)
    except Exception:
        # LinAlgError: matrices that are not square
        return None

    if not np.isclose(power, power[0]).all():
        return None
    # if all rows are the same! -> steady state!
    return np.array([power[0]])
