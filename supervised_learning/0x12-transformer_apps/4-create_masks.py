#!/usr/bin/env python3
""" Transformer Applications - Create Masks """

import tensorflow.compat.v2 as tf

def create_masks(inputs, target):
    """
    Method:
    -------
        Creates all masks for training/validation.

    Parameters:
    -----------
        inputs (tf.Tensor of shape (batch_size, seq_len_in)):
        contains the input sentence.

        target (tf.Tensor of shape (batch_size, seq_len_out)):
        contains the target sentence

    Returns:
        encoder_mask (tf.Tensor of shape (batch_size, 1, 1, seq_len_in)):
        `padding mask` to be applied in the encoder.

        combined_mask (tf.Tensor of shape (batch_size,
                                           1,
                                           seq_len_out,
                                           seq_len_out):
        used in the 1st attention block in the decoder to pad and mask future
        tokens in the input received by the decoder. It takes the maximum
        between a look ahead mask and the decoder target padding mask.

        decoder_mask (tf.Tensor  of shape (batch_size, 1, 1, seq_len_in)):
        `padding mask` used in the 2nd attention block in the decoder.


    **  This function should only use tensorflow operations in order to **
    **            properly function in the training step                **

    """
    def create_padding_mask(seq):
        """ """
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions to add the padding to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    def create_look_ahead_mask(size):
        """ """
        # tf.linalg.bandPart (a, numLower, numUpper)
        # Parameters:
        # a: it is tf.tensor to be passed.
        # numLower: Number of subdiagonals to keep.
        #           If negative, keep entire lower triangle.
        # numUpper: Number of subdiagonals to keep.
        #           If negative, keep entire upper triangle.
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)


    batch_size, seq_len_out = target.shape

    encoder_mask = create_padding_mask(inputs)

    dec_target_padding_mask = create_padding_mask(target)
    look_ahead_mask = create_look_ahead_mask(seq_len_out)

    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    decoder_mask = create_padding_mask(inputs)

    return encoder_mask, combined_mask, decoder_mask
