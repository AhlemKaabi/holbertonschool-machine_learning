#!/usr/bin/env python3
"""
    One Hot
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Method:
        That converts a label vector into a one-hot matrix.

    Parameters:
        @labels: vector

    Returns:
         the one-hot matrix
    """
    # https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical

    return K.utils.to_categorical(labels,
                                  num_classes=classes,
                                  dtype='float32')
