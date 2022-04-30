#!/usr/bin/env python3
""" RNN - GRU Cell  """
import numpy as np


class GRUCell():
    """ Represents a gated recurrent unit GRU """

    def __init__(self, i, h, o):
        """
        constructor
            initializes the weights and biases of the RNN

        Parameters:
            i: the dimensionality of the data
            h: the dimensionality of the hidden state
            o: the dimensionality of the outputs

        ** More details:

        public instance attributes Wz, Wr, Wh, Wy, bz, br, bh, by
        that represent the weights and biases of the cell

            - Wz and bz are for the update gate
            - Wr and br are for the reset gate
            - Wh and bh are for the intermediate hidden state
            - Wy and by are for the output

        The weights  initialized using a random normal distribution in the
        order listed above.

        The weights will be used on the right side for matrix multiplication.

        The biases initialized as zeros.

        **

        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bh = np.zeros((1, h))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
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

        **
        The output of the cell should use a softmax activation function
        **
        """
        # before the  conclusion part
        # http://colah.github.io/posts/2015-08-Understanding-LSTMs/
        combine_r_z = np.concatenate((h_prev, x_t), axis=1)

        # reset gate
        r = np.matmul(combine_r_z, self.Wr) + self.br
        r = 1 / (1 + np.exp(-r))

        # update gate
        z = np.matmul(combine_r_z, self.Wz) + self.bz
        z = 1 / (1 + np.exp(-z))

        # intermediate hidden state
        combine_h = np.concatenate((r * h_prev, x_t), axis=1)

        intermediate = np.matmul(combine_h, self.Wh) + self.bh

        h_next = ((1 - z) * h_prev) + (z * np.tanh(intermediate))

        # output
        y = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, y
