#!/usr/bin/env python3
"""
    Policy Gradients
"""

import numpy as np

def policy(matrix, weight):
    """
    ----- Simple Policy function -----
    Method:
    -------
        Computes to policy with a weight of a matrix,
        using softmax.

    Parameters:
    -----------
        matrix: current observation of the environment.
        weight: matrix of random weight.

    Returns:
    --------
        Weighted policy.
    """
    exp = np.exp(matrix.dot(weight))
    # return the probability of the actions 0 and 1.
    return exp / np.sum(exp)



def softmax_grad(softmax):
    """Find the softmax gradient"""
    s = softmax.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)

def policy_gradient(state, weight):
    """
    ----- Compute the Monte-Carlo policy gradient  -----
    Method:
    -------
        Computes the Monte-Carlo policy gradient based on
        a state and a weight matrix.

    Parameters:
    -----------
        matrix: current observation of the environment.
        weight: matrix of random weight.

    Returns:
    --------
        action, gradient: The action and the gradient (in this order).
    """
    # action Probability
    policy_prob = policy(state, weight)
    # goal: gradient of the policy in state s taking an action a

    # take an action
    action = np.random.choice(len(policy_prob[0]), p=policy_prob[0])

    # softmax gradient
    dsoftmax = softmax_grad(policy_prob)[action, :]

    # gradient of the log
    dlog = dsoftmax / policy_prob[0, action]

    # grade
    gradient = np.dot(state.T, dlog[None, :])

    return action, gradient
