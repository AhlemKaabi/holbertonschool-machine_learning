#!/usr/bin/env python3
""" Deep RNN  """
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Method:
        Performs forward propagation for a deep RNN.

    Parameters:
        rnn_cells (list of RNNCell instances) (length l):
            that will be used for the forward propagation.

        X (given as a numpy.ndarray of shape (t, m, i)):
            the data to be used:
            - t: the maximum number of time steps.
            - m: the batch size.
            - i: the dimensionality of the data.

        h_0(given as a numpy.ndarray of shape (l, m, h)):
            the initial hidden state.
            - h: the dimensionality of the hidden state

    Returns: (H, Y)
        - H: containing all of the hidden states
        - Y: containing all of the outputs
    """
    t, m, _ = X.shape
    L, _, h = h_0.shape
    # output shape of the inital hidden state (l, m, h) + for all steps + 1

    H = np.zeros((t + 1, L, m, h))
    H[0, :, :, :] = h_0
    Y = []

    for step in range(t):
        for layer in range(L):
            if layer == 0:
                # inital input for each layer and each step!
                h_next, y = rnn_cells[layer].forward(H[step, layer], X[step])
            else:
                h_next, y = rnn_cells[layer].forward(H[step, layer], h_next)
            H[step + 1, layer] = h_next
        # outputs of a layer
        Y.append(y)

    Y = np.array(Y)

    return H, Y
