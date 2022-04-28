#!/usr/bin/env python3
""" RNNs """
import numpy as np


class RNNCell():
    """Represents a cell of a simple RNN"""

    def __init__(self, i, h, o):
        """
        constructor
            initializes the weights and biases of the RNN

        Parameters:
            i: the dimensionality of the data
            h: the dimensionality of the hidden state
            o: the dimensionality of the outputs
        """
        # ** weights and biases of the cell **

        # Wh and bh are for the concatenated hidden state and input data
        # Wy and by are for the output

        # weights will be used on the right side for matrix multiplication
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Method:
            Performs forward propagation for one time step.

        Parameters:
            h_prev(numpy.ndarray of shape (m, h)):
                containing the previous hidden state.
                - m is the batch size for the data

            x_t(numpy.ndarray of shape (m, i)):
                contains the data input for the cell.

        Returns:
            h_next: the next hidden state.
            y: the output of the cell.
        """
        h_next = np.matmul(np.concatenate((h_prev, x_t), axis=1),
                           self.Wh) + self.bh

        h_next = np.tanh(h_next)

        y = np.matmul(h_next, self.Wy) + self.by
        # output of the cell should use a softmax activation function
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, y
