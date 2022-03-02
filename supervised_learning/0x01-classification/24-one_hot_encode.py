#!/usr/bin/env python3
"""
    One-Hot Encode
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Method:
        converts a numeric label vector into a one-hot matrix

    Args:
        Y(numpy.ndarray), shape (m,):
            containing numeric class labels
            - m: number of examples

        classes: maximum number of classes found in Y

    Return:
        A one-hot encoding of Y with shape (classes, m),
          or None on failure
    """
    if not isinstance(classes, int) or classes < 1:
        return None

    if not isinstance(Y, np.ndarray):
        return None
    # https://www.kite.com/python/answers/how-to-do-one-hot-encoding-with-numpy-in-python
    data = np.array(Y)
    # classes is larger than largest element in Y
    if classes > data.max():
        return None
    shape = (data.size, data.max() + 1)
    one_hot = np.zeros(shape)
    rows = np.arange(data.size)
    one_hot[rows, data] = 1
    return one_hot.T