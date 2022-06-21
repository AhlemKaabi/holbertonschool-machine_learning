#!/usr/bin/env python3
"""
    Load the Environment
"""
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Method:
    -------
        loads the pre-made FrozenLakeEnv evnironment from OpenAI’s gym

    Parameters:
    -----------
        desc(list or None): list of lists containing a custom description of
          the map to load for the environment.

        map_name(string or None): string containing the pre-made map to load.

        is_slippery(boolean): determine if the ice is slippery.

    Returns:
     --------
          the environment.

     ** If both desc and map_name are None                     **
     ** the environment will load a randomly generated 8x8 map **
    """

    return gym.make("FrozenLake-v1", is_slippery=is_slippery,
                    desc=desc, map_name=map_name)
