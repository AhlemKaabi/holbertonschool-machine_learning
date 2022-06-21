#!/usr/bin/env python3
"""
    Epsilon Greedy
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Method:
     -------
         Uses epsilon-greedy to determine the next action.

    Parameters:
    -----------
        Q(numpy.ndarray): containing the q-table
        state: the current state
        epsilon: the epsilon to use for the calculation
    Returns:
    --------
        The next action index.
    """
    # https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56
    # Epsilon Greedy Action Selection
    # https://www.baeldung.com/cs/epsilon-greedy-q-learning#2-epsilon-greedy-action-selection
    # Q(state_space_size, action_space_size)
    if np.random.uniform(0, 1) < epsilon:
        # Explore: select a random action
        index = np.random.randint(Q.shape[1])
    else:
        # Exploit: select the action with max value (future reward)
        index = np.argmax(Q[state, :])
    return index
