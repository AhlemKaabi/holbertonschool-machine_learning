#!/usr/bin/env python3
"""
    Gradient Descent with L2 Regularization
"""
import numpy as np


def tanh(Z):
    """
    Method:
        Computes the Hyperbolic Tagent of Z elemnet-wise.

    Parameters:
        @Z (array): output of affine transformation.

    Returns:
        A (array): post activation output.
    """
    A = np.tanh(Z)
    return A


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Method:
        updates the weights and biases of a neural network using
          gradient descent with L2 regularization.

    Parameters:
        @Y: a one-hot that contains the correct labels for the data.
        @weights: a dictionary of the weights and biases.
        @cache: a dictionary of the outputs of each layer of the neural network
        @alpha: the learning rate
        @lambtha: the L2 regularization parameter
          @L: the number of layers of the network

    Returns:
        cost of the network accounting for L2 regularization
    """
    # Number of trainings
    m = Y.shape[1]
    # Initialization for backpropagation algorithm
    dZ = cache['A' + str(L)] - Y

    for i in range(L, 1, -1):
        # dW is the Derivative of the cost function w.r.t W
        # of the current layer
        dW = (np.matmul(dZ, cache['A' + str(i - 1)].T)) / m
        # db is the Derivative of the cost function w.r.t b
        # of the current layer
        db = (np.sum(dZ, axis=0, keepdims=True)) / m
        # dA is the Derivative of the cost function w.r.t A
        # (cache) of the current layer
        dA = 1 - np.square(cache['A' + str(i - 1)])
        # dZ is the Derivative of the cost function w.r.t Z
        # of the current layer
        dZ = np.multiply(np.matmul(weights['W' + str(i)].T, dZ), dA)
        # Updating weight matrix and the bias vector for
        # each layer
        reg_term = 1 - ((alpha * lambtha) / m)
        weights['W' + str(i)] = reg_term * weights['W' + str(i)] - alpha * dW
        weights['b' + str(i)] = weights['b' + str(i)] - alpha * db
