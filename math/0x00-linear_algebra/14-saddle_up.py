#!/usr/bin/env python3
"""
    matrix multiplication.
"""
import numpy as np


def np_matmul(mat1, mat2):
    """
        matrix multiplication.
    """

    A = np.array(mat1)
    B = np.array(mat2)
    C = A.dot(B)
    return C
