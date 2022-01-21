#!/usr/bin/env python3
"""
    Forward Propagation with Dropout.
"""
import numpy as np


def dropout_matrices(weights, m, keep_prob, L):
    np.random.seed(1)
    D = {}

    for l in range(1, L + 1):
        # initialize the random values for the dropout matrix
        D[str(l)] = np.random.rand(weights['W' + str(l)].shape[0], m)
        # Convert it to 0/1 to shut down neurons corresponding to each element
        D[str(l)] = (D[str(l)] < keep_prob).astype(int)
        assert(D[str(l)].shape == (weights['W' + str(l)].shape[0], m))
    return D

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
    # get number of examples
    m = X.shape[1]
    # https://tinyurl.com/56eyn35t
    D = dropout_matrices(weights, m, keep_prob, L)
    output = {}
    A = X
    output['A0'] = A
    for i in range(1, L + 1):
        Z = np.matmul(weights['W' + str(i)], A) + weights['b' + str(i)]
        if i == L:
            A = np.exp(Z)/np.sum(np.exp(Z))
        else:
            A = (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z))
        # d = (np.random.rand(A.shape[0], A.shape[1]) < keep_prob).astype(int)
        A = np.multiply(A, D[str(i)])
        A /= keep_prob
        output['A' + str(i)] = A
        output['D' + str(i)] = D[str(i)]
    return output

