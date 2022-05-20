#!/usr/bin/env python3
""" Self Attention  """
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
        Calculate the attention for machine based on the paper:
        NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO A LIGN AND TRANSLATE
    """

    def __init__(self, units):
        """
        Method:
        -------
            Class constructor

        Parameters:
        -----------
            units (integer): representing the number of hidden units
            in the alignment model
        """
        # Dense layer with units units, to be applied to the previous decode
        # hidden state
        self.W = tf.keras.layers.Dense(units)
        # Dense layer with units units, to be applied to the encoder hidden
        # states
        self.U = tf.keras.layers.Dense(units)
        # Dense layer with 1 units, to be applied to the tanh of the sum of
        # the outputs of W and U
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Parameters:
        -----------
            s_prev (tensor of shape(batch, units)): the previous decoder hidden
            state

            hidden_states (tensor of shape(batch, input_seq_len, units)):
            containing the outputs of the encoder.

        Returns: (context, weights)
        --------
            context (tensor of shape (batch, units)): that contains the context
            vector for the decoder.

            weights (tensor of shape (batch, input_seq_len, 1)): that contains
            the attention weights.
        """
        # https://www.youtube.com/watch?v=06r6kp7ujCA&list=PLgtf4d9zHHO8p_zDKstvqvtkv80jhHxoE&index=2
        new_s_prev = tf.expand_dims(s_prev, axis=1)
        score = self.V(tf.nn.tanh(self.W(new_s_prev) + self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights
