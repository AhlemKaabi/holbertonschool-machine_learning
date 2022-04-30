#!/usr/bin/env python3
""" Bidirectional RNN  """
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Method:
        Performs forward propagation for a bidirectional RNN.

    Parameters:
        bi_cell(BidirectinalCell):
            will be used for the forward propagation

        X: the data to be used
            - t: the maximum number of time steps
            - m: the batch size
            - i: the dimensionality of the data

        h_0(given as a numpy.ndarray of shape (m, h)):
            the initial hidden state in the forward direction.
            - h is the dimensionality of the hidden state

        h_t:(given as a numpy.ndarray of shape (m, h)):
            the initial hidden state in the backward direction.

    Returns: (H, Y)
        - H: concatenated hidden states
        - Y: containing all of the outputs
    """
    t, m, _ = X.shape
    m, h = h_0.shape
    H = np.zeros((t, m, h * 2))
    h_prev = np.zeros((t, m, h))
    h_next = np.zeros((t, m, h))
    forward_X = h_0
    backword_X = h_t

    for step in range(t):
        # forward pass
        h_next[step] = bi_cell.forward(forward_X, X[step])
        # backword pass
        h_prev[t - step - 1] = bi_cell.backward(backword_X, X[t - step - 1])
        forward_X = h_next[step]
        backword_X = h_prev[t - step - 1]
    # H shape (t, m, 2 * h)
    H = np.concatenate((h_prev, h_next), axis=2)
    Y = bi_cell.output(H)

    return H, np.array(Y)
