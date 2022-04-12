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
    # try:
    #     power = np.linalg.matrix_power(P, 100)
    # except Exception:
    #     # LinAlgError: matrices that are not square
    #     return None

    # if not np.isclose(power, power[0]).all():
    #     return None
    # if all rows are the same! -> steady state!
    # return np.array([power[0]])

    # another way to slove this: is to use this formula pi * P = pi
    # where pi is the steady state ==> use the eigenvector equation!
    # where the eigenvalue = 1

	# get the eigenvectors and eigenvalues
    eigvals, eigvecs = np.linalg.eig(P.T)


    # get the eigenvalue == 1
    # Returns a boolean array where two arrays are element-wise equal
    # within a tolerance
    isclose = np.isclose(eigvals)
    # the indices of array elements that are non-zero
    eigval_1 = np.argwhere(isclose)

    if len(eigval_1) != 1:
        # not a regular transition matrix
        # should not be different than 1
        return None

    # get the teady state from the eigenvectors matrix
    steady = eigvecs[:, eigval_1[0]]
    # normalization
    steady /= np.sum(eigvecs[:, eigval_1[0]])

    if 0 in steady:
        # => not a regular markov chain
        return None

    return steady.T
