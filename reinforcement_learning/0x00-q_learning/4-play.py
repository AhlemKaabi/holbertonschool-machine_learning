#!/usr/bin/env python3
"""
    Play
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Method:
    -------
        That has the trained agent play an episode.

     Parameters:
     -----------
        env: the FrozenLakeEnv instance
        Q(numpy.ndarray) containing the Q-table
        max_steps: the maximum number of steps in the episode

    Returns:
    --------
        Returns: the total rewards for the episode

    ** Each state of the board should be displayed via the console **
    ** You should always exploit the Q-table                       **
    """
    state = env.reset()
    env.render()

    for step in range(max_steps):

        # Take the action (index) that have the maximum expected future
        # reward given that state
        action = np.argmax(Q[state, :])

        new_state, reward, done, info = env.step(action)

        env.render()

        if done:
            env.render()
            break
        state = new_state
    return reward
