#!/usr/bin/env python3
"""
    One-Hot Decode
"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Method:
        converts a one-hot matrix into a vector of labels.

    Args:
        one_hot(numpy.ndarray), shape (classes, m)
            - classes: the maximum number of classes
            - m: the number of examples

    Return:
        (numpy.ndarray), shape (m, )
        containing the numeric labels for each example,
        or None on failure
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None
    return np.argmax(one_hot, axis=0)
