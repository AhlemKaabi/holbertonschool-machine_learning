#!/usr/bin/env python3
""" Positional Encoding """
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Method:
    -------
        Calculates the positional encoding for a transformer

    Parameters:
    -----------
        max_seq_len(integer): maximum sequence length, containign the input
        embeddings.

        dm: model depth (in the paper attention is all you need: 512)

    Returns:
    --------
        a numpy.ndarray of shape (max_seq_len, dm) containing the positional
        encoding vectors
    """
    pos_vec = np.zeros((max_seq_len, dm))

    for position in range(max_seq_len):
        for i in range(0, dm, 2):
            # i // 2 : get every index!
            div = np.power(10000, (2 * i // 2) / dm)
            pos_vec[position, i] = (np.sin(position / div))
            pos_vec[position, i + 1] = (np.cos(position / div))

    return pos_vec
