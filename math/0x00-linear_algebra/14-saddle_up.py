#!/usr/bin/env python3
"""
    matrix multiplication.
"""


def np_matmul(mat1, mat2):
    """
        matrix multiplication.
    """
    import numpy as np

    A = np.array(mat1)
    B = np.array(mat2)
    C = A.dot(B)
    return C
