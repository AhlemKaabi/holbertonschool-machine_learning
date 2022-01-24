#!/usr/bin/env python3
"""
    Gradient Descent with Dropout
"""
from re import I
import numpy as np


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

    # the last Layer uses the softmax activation function
    softmax = cache['A' + str(L)]
    dZ = softmax - Y
    # print(" init dZ shape", dZ.shape)
    # print(cache['D2'].shape)
    # print(cache['D1'].shape)

    for i in range(L, 0, -1):
        # print("this is layer", l)

        W = weights['W' + str(i)]
        # print("W shape", W.shape)

        b = weights['b' + str(i)]

        tanh_A = cache['A' + str(i - 1)]
        # print("tanh_A shape", tanh_A.shape)
        dW = (np.matmul(dZ, tanh_A.T)) / m
        # print("dW shape", dW.shape)

        db = (np.sum(dZ, axis=0, keepdims=True)) / m
        # if l == L:
        #
        #     # dA = 1 - np.square(cache['A' + str(l)])
        #     print("dA shape", dA.shape)

        if i > L:
            # A3 . D2 / A2 . D1
            dA = 1 - np.square(cache['A' + str(i + 1)])
            current_D = cache['D' + str(i)]
            # print("current_D shape", current_D.shape)

            dA = (dA * current_D) / keep_prob
            # print(" l != L + dropout dA shape", dA.shape)
        else:
            dA = 1 - np.square(tanh_A)

        # dA = (dA * current_D) / keep_prob
        dZ = np.matmul(W.T, dZ) * dA
        # print("dZ shape", dZ.shape)

        weights['W' + str(i)] = W - (alpha * dW)
        weights['b' + str(i)] = b - (alpha * db)
