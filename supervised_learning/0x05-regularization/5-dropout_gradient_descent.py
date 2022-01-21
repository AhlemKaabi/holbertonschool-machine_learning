#!/usr/bin/env python3
"""
    Gradient Descent with Dropout
"""
import numpy as np


def dropout_matrices(weights, m, keep_prob, L):
    """
    Method:
        create dropout matrices

    Parameters:
        @weights: is a dictionary of the weights and biases of
              the neural network
        @m: number of examples
        @L: the number of layers in the network.
        @keep_prob: the probability that a node will be kept.

    Returns:
        dictionary containing the dropouts
    """
    np.random.seed(1)
    D = {}

    for i in range(1, L + 1):
        # initialize the random values for the dropout matrix
        D[str(i)] = np.random.rand(weights['W' + str(i)].shape[0], m)
        # Convert it to 0/1 to shut down neurons corresponding to each element
        D[str(i)] = (D[str(i)] < keep_prob).astype(int)
        assert(D[str(i)].shape == (weights['W' + str(i)].shape[0], m))
    return D

def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Method:
        updates the weights of a neural network with Dropout
          regularization using gradient descent:

    Parameters:
        @Y:  one-hot numpy.ndarray  contains the correct
              labels for the data.
        @weights: a dictionary of the weights and biases
              of the neural network.
        @cache: a dictionary of the outputs and dropout masks of
              each layer of the neural network.
          @alpha: the learning rate
        @keep_prob: the probability that a node will be kept.
        @L: the number of layers of the network.

    """
    # get number of examples
    m = Y.shape[1]

    D = dropout_matrices(weights, m, keep_prob, L)

    A = cache['A' + str(L)]
    # dA for output layer
    dA = 1 - np.square(cache['A' + str(L - 1)])
    dA = np.matmul(dA, D[str(L)].T)
    dA /= keep_prob


    cache["A" + str(L)] = dA

    for l in range(L - 1, 0, -1):
        current_activation = cache['A' + str(l)]

        cache["dA" + str(l - 1)] = np.matmul(current_activation, D[str(l)].T)

        cache["dA" + str(l - 1)] /= keep_prob


