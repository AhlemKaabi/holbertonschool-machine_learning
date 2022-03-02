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
    # classes is less than 2
    if not isinstance(classes, int) or classes < 2:
        return None

    if not isinstance(Y, np.ndarray):
        return None



    # https://www.kite.com/python/answers/how-to-do-one-hot-encoding-with-numpy-in-python
    try:
        data = np.array(Y)
        shape = (data.size, data.max() + 1)
        one_hot = np.zeros(shape)
        rows = np.arange(data.size)
        one_hot[rows, data] = 1
        return one_hot.T
    except Exception:
        # classes is larger than largest element in Y
        # classes is smaller than largest element in Y
        return None