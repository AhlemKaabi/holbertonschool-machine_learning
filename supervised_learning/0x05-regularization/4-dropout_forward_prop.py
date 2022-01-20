#!/usr/bin/env python3
"""
    Forward Propagation with Dropout.
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Method:
         conducts forward propagation using Dropout.

    Parameters:
        @X: containing the input data for the network.
        @weights: is a dictionary of the weights and biases of
              the neural network
          @L: the number of layers in the network.
          @keep_prob: the probability that a node will be kept.

    Returns:
        a dictionary containing the outputs of each layer and
        the dropout mask used on each layer
    """
    # https://tinyurl.com/56eyn35t
    output = {}
    A = X
    output['A0'] = X
    for i in range(1, L + 1):
        Z = np.matmul(weights['W' + str(i)], A) + weights['b' + str(i)]
        if i == L:
            A = np.exp(Z)/np.sum(np.exp(Z))
        else:
            A = np.tanh(Z)
        d = (np.random.rand(A.shape[0], A.shape[1]) < keep_prob).astype(int)
        A = np.multiply(A, d)
        A /= keep_prob
        output['A' + str(i)] = A
        output['D' + str(i)] = d
    return output
