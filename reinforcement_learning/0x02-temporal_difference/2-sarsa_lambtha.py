#!/usr/bin/env python3
"""
    TD-Lambda
"""
from operator import ne
import numpy as np

def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Method:
    -------
        Performs SARSA(λ)

    Parameters:
    -----------
        *env: the openAI environment instance
        *Q(numpy.ndarray of shape (s,a)): containing the Q table
        *lambtha: the eligibility trace factor
        *episodes: the total number of episodes to train over
        *max_steps: the maximum number of steps per episode
        *alpha: the learning rate
        *gamma: the discount rate
        *epsilon: the initial threshold for epsilon greedy
        *min_epsilon: the minimum value that epsilon should decay to
        *epsilon_decay: the decay rate for updating epsilon between episodes

    Returns:
    --------
        Q: the updated Q table
    """
    for _ in range(episodes):
        eligibility_trace = np.zeros(Q.shape)
        # initialize state and action
        state = env.reset()
        action = 0
        # repeat for each step
        for _ in range(max_steps):
            # take action, observe reward and new_state
            new_state, reward, done, _ = env.step(action)

            # choose new_action from new_state using policy from Q(e-greedy)
            if np.random.uniform(0, 1) < epsilon:
                new_action = env.action_space.sample()
            else:
                new_action = np.argmax(Q[state, :])

            delta = reward + gamma * Q[new_state, new_action] - Q[state, action]
            eligibility_trace[state, action] += 1.0
            Q[state, action] += alpha * delta * eligibility_trace[state, action]
            eligibility_trace[state, action] *= lambtha * gamma
            if done:
                break
            # Update state and action
            state = new_state
            action = new_action
    return Q
