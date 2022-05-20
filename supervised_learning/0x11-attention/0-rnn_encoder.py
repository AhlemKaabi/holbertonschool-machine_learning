#!/usr/bin/env python3
""" RNN Encoder  """
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """ Encode for machine translation """

    def __init__(self, vocab, embedding, units, batch):
        """
        constructor:
        ------------
            Initialize the model

        Parameters:
        -----------
            vocab (integer): representing the size of the input vocabulary

            embedding (integer):  representing the dimensionality of
                the embedding vector

            units (integer): representing the number of hidden units in the
                RNN cell

            batch (integer): representing the batch size
        """
        self.batch = batch
        self.units = units
        # a keras Embedding layer that converts words from the vocabulary into
        # an embedding vector
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        # Gated Recurrent Unit  a keras GRU layer with #units
        self.gru = tf.keras.layers.GRU(self.units,
                                       kernel_initializer="glorot_uniform",
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """
        Method:
        -------
            Initializes the hidden states for the RNN cell to a
              tensor of zeros.

        Returns:
        --------
              tensor (shape (batch, units))
                containing the initialized hidden states.
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Parameters:
        -----------
            x (tensor of shape(batch, input_seq_len)): containing the input
            to the encoder layer as word indices within the vocabulary.

            initial (tensor of shape(batch, units)): containing the initial
            hidden state.

        Returns: (outputs, hidden)
        --------
            outputs (tensor of shape (batch, input_seq_len, units)):
            containing the outputs of the encoder.

            hidden (tensor of shape (batch, units)): containing the last
            hidden state of the encoder.

        """
        input = self.embedding(x)
        # self.embedding() Output:
        # 3D tensor with shape: (batch_size, input_length, output_dim).
        whole_sequence_output, final_state = self.gru(input,
                                                      initial_state=initial)
        # https://keras.io/api/layers/recurrent_layers/gru/
        return whole_sequence_output, final_state
