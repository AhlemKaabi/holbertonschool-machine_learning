#!/usr/bin/env python3
"""
    Initialize Q-table
"""
import numpy as np


def q_init(env):
    """
    Method:
    -------
        initializes the Q-table

    Parameters:
    -----------
        env: the FrozenLakeEnv instance

    Returns:
    --------
        the Q-table as a numpy.ndarray of zeros
    """
    # https://www.youtube.com/watch?v=QK_PP_2KgGE
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n
    q_table = np.zeros((state_space_size, action_space_size))
    return q_table
