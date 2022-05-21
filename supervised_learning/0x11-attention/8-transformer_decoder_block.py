#!/usr/bin/env python3
""" Transformer Decoder Block  """
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
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
        super(DecoderBlock, self).__init__()
        #  the first MultiHeadAttention layer
        self.mha1 = MultiHeadAttention(dm, h)
        # the second MultiHeadAttention layer
        self.mha2 = MultiHeadAttention(dm, h)
        # the hidden dense layer with hidden units and relu activation
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        # the output dense layer with dm units
        self.dense_output = tf.keras.layers.Dense(units=dm)
        # the first layer norm layer, with epsilon=1e-6
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # the second layer norm layer, with epsilon=1e-6
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # the third layer norm layer, with epsilon=1e-6
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # the first dropout layer
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        # the second dropout layer
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        #  the third dropout layer
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Parameters:
        -----------
            x (tensor of shape (batch, target_seq_len, dm)): containing the
            input to the decoder block

            encoder_output: a tensor of shape (batch, input_seq_len, dm)
            containing the output of the encoder

            training: a boolean to determine if the model is training

            look_ahead_mask: the mask to be applied to the first multi
            head attention layer

            padding_mask: the mask to be applied to the second multi
            head attention layer

        Returns:
        --------
            (tensor of shape (batch, target_seq_len, dm))
            containing the block’s output
        """
        Q = x
        K = x
        V = x
        attn1, _ = self.mha1(Q, K, V, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        norm_1 = self.layernorm1(attn1 + x)

        attn2, _ = self.mha2(
            norm_1,
            encoder_output,
            encoder_output,
            padding_mask
        )
        attn2 = self.dropout2(attn2, training=training)
        norm_2 = self.layernorm2(attn2 + norm_1)

        feedforward = tf.keras.Sequential([self.dense_hidden, self.dense_output])
        fforward = feedforward(norm_2)
        # feedforward = self.dense_hidden(norm_1)
        # feedforward = self.dense_output(feedforward)

        fforward = self.dropout3(fforward, training=training)
        norm_3 = self.layernorm3(fforward + norm_2)
        return norm_3
