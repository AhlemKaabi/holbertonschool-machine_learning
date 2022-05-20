#!/usr/bin/env python3
""" Scaled Dot Product Attention  """
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Method:
    -------
        Calculates the scaled dot product attention.

    Parameters:
    -----------
        Q (tensor its last two dimensions as (..., seq_len_q, dk)): containing
        the query matrix.

        K (tensor its last two dimensions as (..., seq_len_v, dk)): containing
        the key matrix.

        V (tensor its last two dimensions as (..., seq_len_l, dk)): containing
        the value matrix.

        ** The preceding dimensions of Q, K, and V are the same **

        mask(tensor broadcast into (..., seq_len_q, seq_len_v)): containing
        the optional mask, or defaulted to None.

        **
          If mask is not None, multiply -1e9 to the mask and add it to the
          scaled matrix multiplication.
        **
    Returns:
    --------
        output (tensor its last two dimensions as (..., seq_len_q, dv))
        containing the scaled dot product attention

        weights (tensor its last two dimensions as (..., seq_len_q, seq_len_v))
        containing the attention weights
    """
    # watch: https://www.youtube.com/watch?v=mMa2PmYJlCo

    # compute matmul layer - > attention scores
    attention_filter = tf.matmul(Q, K, transpose_b=True)

    # dimension of key vector -> cast to prevent the error
    scaling = tf.cast(tf.shape(K)[-1], tf.float32)

    # compute similarity
    scale_layer = attention_filter / tf.math.sqrt(scaling)

    if mask is not None:
        scale_layer += (mask * -1e9)
    # attention weights
    final_attention_filter = tf.nn.softmax(scale_layer)

    # scaled dot product attention
    filtered_value = tf.matmul(final_attention_filter, V)
    return filtered_value, final_attention_filter
