#!/usr/bin/env python3
""" RNN - LSTM Cell """
import numpy as np


class LSTMCell():
    """ Represents LSTM unit """

    def __init__(self, i, h, o):
        """
        Constructor
            initializes the weights and biases of the RNN

        Parameters:
            i: the dimensionality of the data
            h: the dimensionality of the hidden state
            o: the dimensionality of the outputs

        ** More details:

        Public instance attributes Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo, by
        that represent the weights and biases of the cell.
            Wf and bf are for the forget gate
            Wu and bu are for the update gate
            Wc and bc are for the intermediate cell state
            Wo and bo are for the output gate
            Wy and by are for the outputs

        The weights initialized using a random normal distribution
        in the order listed above

        The weights will be used on the right side for matrix multiplication

        The biases initialized as zeros
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Method:
            Perform forward propagation for one time step.

        Parameters:
            h_prev(numpy.ndarray of shape (m, h)):
                containing the previous hidden state.
                - m is the batch size for the data

            c_prev(numpy.ndarray of shape (m, h)):
                containing the previous cell state

            x_t(numpy.ndarray of shape (m, i)):
                contains the data input for the cell.

        Returns:
            h_next: the next hidden state.
            c_next: the next cell state.
            y: the output of the cell.

        **
        The output of the cell should use a softmax activation function
        **
        """
        combine = np.concatenate((h_prev, x_t), axis=1)

        # forget gate
        f = np.matmul(combine, self.Wf) + self.bf
        f = (1 / (1 + np.exp(-f)))

        # update gate
        u = np.matmul(combine, self.Wu) + self.bu
        u = (1 / (1 + np.exp(-u)))

        #  intermediate cell state
        c = np.matmul(combine, self.Wc) + self.bc
        c = np.tanh(c)

        # output gate
        o = np.matmul(combine, self.Wo) + self.bo
        o = (1 / (1 + np.exp(-o)))
        # next cell state
        c_next = (u * c) + (f * c_prev)
        # next hidden state
        h_next = o * np.tanh(c_next)
        # outputs
        y = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, c_next, y
