#!/usr/bin/env python3
""" RNN - Bidirectional Cell Backward """
import numpy as np


class BidirectionalCell():
    """ Represents bidirectional cell of an RNN """

    def __init__(self, i, h, o):
        """
        constructor
            initializes the weights and biases of the RNN

        Parameters:
            i: the dimensionality of the data
            h: the dimensionality of the hidden state
            o: the dimensionality of the outputs

        ** More details:

          Public instance attributes Whf, Whb, Wy, bhf, bhb, by
          that represent the weights and biases of the cell.

            - Whf and bhf are for the hidden states in the forward direction.
            - Whb and bhb are for the hidden states in the backward direction.
            - Wy and by are for the outputs.

        The weights initialized using a random normal distribution in
        the order listed above.should be

        The weights will be used on the right side for matrix multiplication.

        The biases initialized as zeros.
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=((2 * h), o))

        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Method:
            Calculates the hidden state in the forward direction
             or one time step

        Parametrs:
            h_prev(numpy.ndarray of shape (m, h)):
                containing the previous hidden state.
                - m is the batch size for the data

            x_t(numpy.ndarray of shape (m, i)):
                contains the data input for the cell.

        Returns:
            h_next: the next hidden state.
        """
        combine = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.matmul(combine, self.Whf) + self.bhf
        h_next = np.tanh(h_next)

        return h_next

    def backward(self, h_next, x_t):
        """
        Method:
            Calculates the hidden state in the backward direction
             or one time step

        Parametrs:
            h_next(numpy.ndarray of shape (m, h)):
                containing the next hidden state.
                - m is the batch size for the data

            x_t(numpy.ndarray of shape (m, i)):
                contains the data input for the cell.

        Returns:
            h_next: the previous hidden state.
        """
        combine = np.concatenate((h_next, x_t), axis=1)

        h_prev = np.matmul(combine, self.Whf) + self.bhf
        h_prev = np.tanh(h_prev)

        return h_prev
