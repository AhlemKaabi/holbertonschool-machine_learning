#!/usr/bin/env python3
"""
	fist-visit Monte-Carlo
"""
import numpy as np

def generate_episode(env, policy, max_steps):
    """
    """
    # initialize a state s
    s = env.reset()
    # print(s)
    episodes_list = []
    for j in range(max_steps):
        # take action following a policy from a state
        action = policy(s)

        # the return of the new step action -> define new state
        s_new, reward, done, info = env.step(action)

        episodes_list.append([s, action, reward, s_new])

        if done:
            break
        s = s_new

    return episodes_list

def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """
    Method:
    -------
        Performs the Monte Carlo algorithm.

    Parameters:
    -----------
        *env: the openAI environment instance
        *V(numpy.ndarray of shape (s,)): the value estimate
        *policy: a function that takes in a state and returns
           the next action to take
        *episodes: the total number of episodes to train over
        *max_steps: the maximum number of steps per episode
        *alpha: the learning rate
        *gamma: the discount rate

    Returns:
    --------
        V: the updated value estimate
    """
    for i in range(episodes):
        episodes_list = generate_episode(env, policy, max_steps)

        episodes_list = np.array(episodes_list, dtype=int)
        # return_G the total discounted reward
        return_G = 0
        for j, step in enumerate(episodes_list[::-1]):
            state, action, reward, new_state = step
            return_G = gamma * return_G + reward
            # following the first visit MC algorithm
            if state not in episodes_list[:i, 0]:
                V[state] = V[state] + alpha * (return_G - V[state])

    return V
