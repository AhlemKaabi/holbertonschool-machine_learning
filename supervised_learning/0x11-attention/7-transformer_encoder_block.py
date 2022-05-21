#!/usr/bin/env python3
""" Transformer Encoder Block  """
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """ create an encoder block for a transformer """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Method:
        -------
            Class constructor

        Parameters:
        -----------
            dm: the dimensionality of the model
            h: the number of heads
            hidden: the number of hidden units in the fully connected layer
            drop_rate: the dropout rate
        """
        super(EncoderBlock, self).__init__()
        # a MultiHeadAttention layer
        self.mha = MultiHeadAttention(dm, h)
        # the hidden dense layer with hidden units and relu activation
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        # the output dense layer with dm units
        self.dense_output = tf.keras.layers.Dense(dm)
        # the first layer norm layer, with epsilon=1e-6
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # the second layer norm layer, with epsilon=1e-6
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # the first dropout layer
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        # the second dropout layer
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Parameters:
        -----------
            x: tensor of shape (batch, input_seq_len, dm)containing the input
                    to the encoder block
            training: boolean to determine if the model is training
            mask: the mask to be applied for multi head attention

        Return:
        -------
            tensor of shape (batch, input_seq_len, dm) with the block’s output
        """
        Q = x
        K = x
        V = x
        attention_output, _ = self.mha(Q, K, V, mask)
        attention_output = self.dropout1(attention_output, training=training)
        norm_1 = self.layernorm1(x + attention_output)
        feedforward = self.dense_hidden(norm_1)
        feedforward = self.dense_output(feedforward)
        feedforward = self.dropout2(feedforward, training=training)
        norm_2 = self.layernorm2(norm_1 + feedforward)
        return norm_2
