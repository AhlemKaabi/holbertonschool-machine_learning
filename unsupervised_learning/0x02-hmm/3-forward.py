#!/usr/bin/env python3
"""
    Hidden Markov Models - The Forward Algorithm
    Goal: joint distribution of Z_k and X_1:k

"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Method to perform the forward algorithm for a hidden markov model.

    Parameters:
        Observation (numpy.ndarray of shape (T,)):
         contains the index of the observation
         - T is the number of observations

        Emission (numpy.ndarray of shape (N, M)):
         containing the emission probability of a specific observation given
         a hidden state
            - Emission[i, j] is the probability of observing j given the hidden
            state i
             - N: the number of hidden states.
             - M: the number of all possible observations.

        Transition (2D numpy.ndarray of shape (N, N)):
         containing the transition probabilities
            - Transition[i, j] is the probability of transitioning from the
            hidden state i to j

        Initial (numpy.ndarray of shape (N, 1)):
         containing the probability of starting in a particular hidden state.

    Returns: P, F, or None, None on failure
        P: the likelihood of the observations given the model
        F (numpy.ndarray of shape (N, T)): containing the forward path
         probabilities
            - F[i, j] is the probability of being in hidden state i at time j
            given the previous observations.
    """

    T = Observation.shape[0]
    N, M = Emission.shape

    forward = np.zeros((N, T))

    for s in range(N):
        forward[s, 0] = Initial[s] * Emission[s, Observation[0]]
    # or => forward[:, 0] = Initial.T * Emission[:, Observation[0]]
    for t in range(1, T):
        for s in range(N):
            # T_mul_E = Transition[:, s] * Emission[s, Observation[t]]
            sum_mul =np.matmul(forward[:, t - 1], Transition[:, s])
            forward[s, t] = sum_mul * Emission[s, Observation[t]]

    P = np.sum(forward[:, T - 1])

    return P, forward
