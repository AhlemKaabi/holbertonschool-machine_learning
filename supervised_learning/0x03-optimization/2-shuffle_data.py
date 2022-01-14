#!/usr/bin/env python3
"""
    Shuffle Data
    shuffling techniques aim to mix up data and can optionally
    retain logical relationships between columns.
"""
import numpy as np



def shuffle_data(X, Y):
    """
    shuffles the data points in two matrices the same way

    Parameters:
        X (numpy.ndarray): of shape(m, nx) to shuffle
            m: number of data points
            nx: number of features
        Y (numpy.ndarray): of shape(m, nx) to shuffle
            m: number of data points
            nx: number of features

    Returns:
         The shuffled X and Y matrices.
    """
    # https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing

    # specify the length of the randomized sequence to be equal to the number of elements

    length = np.random.permutation(len(X))
    # !! type(length) = <class 'numpy.ndarray'> !!
    # Use the randomized sequence length
    # as an index for both arrays and returned them.

    X_sh = X[length]
    Y_sh = Y[length]

    return X_sh, Y_sh