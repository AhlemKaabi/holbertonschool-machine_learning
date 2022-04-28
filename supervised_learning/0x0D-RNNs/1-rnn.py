#!/usr/bin/env python3
""" RNNs """
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Method to perform forward propagation for a simple RNN.

    Parameters:
        rnn_cell (instance of RNNCell ):
            will be used for the forward propagation

        X: (given as a numpy.ndarray of shape (t, m, i)):
        - t: the maximum number of time steps
        - m:the batch size
        - i:the dimensionality of the data

        h_0(given as a numpy.ndarray of shape (m, h)):
            the initial hidden state
    Returns: H, Y
        - H(numpy.ndarray): all of the hidden states.
        - Y(numpy.ndarray): all of the outputs.
    """
    h_next = h_0.copy()
    H = [h_next]

    Y = []

    for t in range(X.shape[0]):
        h_next, y = rnn_cell.forward(h_next, X[t])
        H.append(h_next)
        Y.append(y)

    return np.array(H), np.array(Y)
