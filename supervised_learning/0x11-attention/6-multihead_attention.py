#!/usr/bin/env python3
""" Multi Head Attention """
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """ Perform multi head attention """

    def __init__(self, dm, h):
        """
        Method:
        -------
            Class constructor

        Parameters:
        -----------
            dm (integer): representing the dimensionality of the model.

            h (integer): representing the number of heads.

        **
        dm is divisible by h
        **
        """
        # Sets the following public instance attributes:

        #     h - the number of heads
        #     dm - the dimensionality of the model
        #     depth - the depth of each attention head
        #     Wq - a Dense layer with dm units,
        #       used to generate the query matrix
        #     Wk - a Dense layer with dm units,
        #       used to generate the key matrix
        #     Wv - a Dense layer with dm units,
        #       used to generate the value matrix
        #     linear - a Dense layer with dm units,
        #       used to generate the attention output
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask):
        """
        Public Instance Method
        Q: tensor shape (batch, seq_len_q, dk) contains input to
            generate the query matrix
        K: tensor shape (batch, seq_len_v, dk) contains input to
            generate the key matrix
        V: tensor shape (batch, seq_len_v, dv) contains input to
            generate the value matrix
        Returns:
            output: tensor with last two dims (..., seq_len_q, dm)
                contains scaled dot product attention
            w: tensor with last three dims
                (..., h, seq_len_q, seq_len_v) contains attention w
        """
        # https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
        batch_size, seq_len, _ = Q.shape
        QWq = self.Wq(Q)
        # QWq.shape: (50, 15, 512)
        KWk = self.Wk(K)
        VWv = self.Wv(V)
        param = (batch_size, seq_len, self.h, self.depth)
        QWq = tf.reshape(QWq, param)
        # QWq.shape: (50, 15, 8, 64)
        QWq = tf.transpose(QWq, perm=[0, 2, 1, 3])
        # QWq.shape: (50, 8, 15, 64)
        KWk = tf.reshape(KWk, param)
        KWk = tf.transpose(KWk, perm=[0, 2, 1, 3])
        VWv = tf.reshape(VWv, param)
        VWv = tf.transpose(VWv, perm=[0, 2, 1, 3])
        filtered_value, attention_filter = sdp_attention(QWq, KWk, VWv, mask)
        # filtered_value.shape: (..., seq_len_q, dv): (50, 8, 15, 64)
        head_output = tf.transpose(filtered_value, perm=[0, 2, 1, 3])
        # head_output.shape: (50, 15, 8, 64)
        concat = tf.reshape(head_output, (batch_size, seq_len, self.dm))
        # concat.shape: (50, 15, 512)
        output = self.linear(concat)
        # output.shape: (50, 15, 512)
        return output, attention_filter

    # [0.06685878 0.06464797 0.06168237 ... 0.07111148 0.06448605
    # 0.05386085]
    #  [0.06893092 0.06967342 0.07954863 ... 0.04898299 0.06076638
    # 0.06906604]