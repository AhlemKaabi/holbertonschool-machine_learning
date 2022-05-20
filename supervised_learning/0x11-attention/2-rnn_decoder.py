#!/usr/bin/env python3
""" RNN Decoder  """
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """ Decode for machine translation """

    def __init__(self, vocab, embedding, units, batch):
        """
        Method:
        -------
            Class constructor

        Parameters:
        -----------
            vocab (integer): the size of the output vocabulary

            embedding (integer): the dimensionality of the embedding vector

            units (integer): number of hidden units in the RNN cell

            batch (integer): batch size
        """
        super(RNNDecoder, self).__init__()
        # a keras Embedding layer that converts words from the vocabulary into
        # an embedding vector
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        # Gated Recurrent Unit:  a keras GRU layer with #units
        self.gru = tf.keras.layers.GRU(units,
                                       kernel_initializer="glorot_uniform",
                                       return_sequences=True,
                                       return_state=True)
        # Dense layer with #vocab units
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(self.units)

    def call(self, x, s_prev, hidden_states):
        """
        Parameters:
        -----------
            x (tensor of shape(batch, 1)): containing the previous word in the
            target sequence as an index of the target vocabulary.

            s_prev (tensor of shape(batch, units)): containing the previous
            decoder hidden state

            hidden_states (tensor of shape(batch, input_seq_len, units)):
            containing the outputs of the encoder

        Returns: (y, s)
        --------
            y(tensor of shape (batch, vocab)): the output word as a one hot
            vector in the target vocabulary

            s(tensor of shape (batch, units)): the new decoder hidden state
        """
        context_vector, weights = self.attention(s_prev, hidden_states)
        # You should concatenate the context vector with x in that order
        x = self.embedding(x)
        # x.shape (batch, 1)
        # context_vector.shape (batch, units)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)
        # output.shape (batch, input_seq_len, units))
        # state.shape (batch, units)
        output = tf.reshape(output, (-1, output.shape[2]))
        # print(output.shape)
        y = self.F(output)

        return y, state
