#!/usr/bin/env python3
"""
    TD-Lambda
"""
import numpy as np

def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """
    Method:
    -------
        Performs the TD(λ) algorithm.

    Parameters:
    -----------
        *env: the openAI environment instance
        *V(numpy.ndarray of shape (s,)): the value estimate
        *policy: a function that takes in a state and returns
           the next action to take
        *lambtha: the eligibility trace factor.
        *episodes: the total number of episodes to train over
        *max_steps: the maximum number of steps per episode
        *alpha: the learning rate
        *gamma: the discount rate

    Returns:
    --------
        V: the updated value estimate
    """
    states = V.shape[0]
    eligibility_trace = np.zeros(states)
    # print(eligibility_trace)
    for _ in range(episodes):
        obs_state = env.reset()
        # print(obs_state)
        for _ in range(max_steps):
            action = policy(obs_state)

            new_obs_state, reward, done, _ = env.step(action)
            # print(new_obs_state)
            delta = reward + gamma * V[new_obs_state] - V[obs_state]

            eligibility_trace[obs_state] += 1.0
            # V[obs_state] += alpha * delta * eligibility_trace[obs_state]
            V += alpha * delta * eligibility_trace
            eligibility_trace *= lambtha * gamma
            if done:
                break
            obs_state = new_obs_state
    return V
