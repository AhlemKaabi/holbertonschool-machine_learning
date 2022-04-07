#!/usr/bin/env python3
"""
    Hidden Markov Models - Markov Chain
"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Method to etermine the probability of a markov chain being in a particular
    state after a specified number of iterations.

    Parameters:
        P ((square 2D numpy.ndarray) of shape (n, n)):
         representing the transition matrix.
        - P[i, j]: the probability of transitioning from state i to state j
        - n : the number of states in the markov chain

        s (numpy.ndarray of shape (1, n)): representing the probability of
         starting in each state.
        t (int) :the number of iterations that the markov chain has been
         through.

    Returns:
        (numpy.ndarray of shape (1, n)) representing the probability of being
        in a specific state after t iterations, or None on failure.
    """
    state_matrix = s

    for _ in range(t):
        next_state_matrix = np.matmul(state_matrix, P)
        state_matrix = next_state_matrix

    return state_matrix
