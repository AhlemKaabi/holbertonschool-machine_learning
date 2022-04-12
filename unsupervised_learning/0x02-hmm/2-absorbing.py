#!/usr/bin/env python3
"""
    Hidden Markov Models - Absorbing Chains

    A Markov chain is an absorbing Markov Chain if

    - It has at least one absorbing state

AND

    - From any non-absorbing state in the Markov chain, it is possible
    to eventually move to some absorbing state (in one or more transitions).

"""
import numpy as np

def get_to_abs_state(abs_states, i, P):
    """
    Method to append the index of non-absorbing states
    """
    culumn = P.T[i]
    for i in range(P.shape[0]):
        if culumn[i] > 0:
            # there is a possibility to move to an absorbing state
            abs_states.append(i)
    return abs_states


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
    # https://math.libretexts.org/Bookshelves/Applied_Mathematics/Applied_Finite_Mathematics_(Sekhon_and_Bloom)/10%3A_Markov_Chains/10.04%3A_Absorbing_Markov_Chains
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return False

    n, m = np.shape(P)

    if n != m:
        return False

    if not np.all(np.isclose(P.sum(axis=1), 1)):
        return False


    # have all or at least one absorbing state
    diag = np.diagonal(P)

    if not np.any( diag == 1):
        return False

    if np.all( diag == 1):
        return True

    absorbing_states = []
    # all absorbing states
    for i in range(len(diag)):
        if diag[i] == 1:
            # save the index
            absorbing_states.append(i)

    for i in range(n):
        if i in absorbing_states:
            absorbing_states = get_to_abs_state(absorbing_states, i, P)

    # for i in range(n):
    #     if i in absorbing_states:
    #         absorbing_states = get_to_abs_state(absorbing_states, i, P)

    absorbing_states = set(absorbing_states) # remove repeated states
    return len(absorbing_states) == n
