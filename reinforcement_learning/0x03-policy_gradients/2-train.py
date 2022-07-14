#!/usr/bin/env python3
"""
    Training - Policy Gradients
"""
import numpy as np
from policy_gradient import policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    ----- Implement the training -----
    Method:
    -------
         Implements a full training.

    Parameters:
    -----------

        env: initial environment
        nb_episodes: number of episodes used for training
        alpha: the learning rate
        gamma: the discount factor


    Returns:
    --------
        all values of the score (sum of all rewards during one episode loop)
    """
    weight = np.random.rand(4, 2)
    scores = []

    for episode in range(nb_episodes):

        # play episode
        state = env.reset()[None, :]
        episodes = []

        while True:
            action, gradient = policy_gradient(state, weight)
            state, reward, done, _ = env.step(action)
            state = state[None, :]
            episodes.append((state, action, reward, gradient))
            if done:
                break
        # count score
        score = 0
        for i in range(len(episodes)):
            # get reward and gradient
            _, _, reward, gradient = episodes[i]
            score += reward

            G = np.sum([ gamma ** episodes[k][2] * episodes[k][2] for k in range(i + 1, len(episodes) + 1)])

            weight += alpha * G * gradient
        scores.append(score)
        print("{}: {}".format(episode, score), end="\r", flush=False)
    return scores
