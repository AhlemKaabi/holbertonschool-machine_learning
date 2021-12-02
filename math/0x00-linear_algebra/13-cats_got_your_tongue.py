#!/usr/bin/env python3
"""
    Concatenates two matrices along a specific axis.
"""


def np_cat(mat1, mat2, axis=0):
    """
        Concatenates two matrices along a specific axis.
    """
    import numpy as np
    new_array = np.concatenate((mat1, mat2), axis=axis)
    return new_array
