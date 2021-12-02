#!/usr/bin/env python3
"""
    element-wise addition, subtraction,
    multiplication, and division.
"""


def np_elementwise(mat1, mat2):
    """
        element-wise addition, subtraction,
        multiplication, and division.
    """
    import numpy as np
    # addition
    add = np.array(np.add(mat1, mat2))
    # subtraction
    sub = np.array(np.subtract(mat1, mat2))
    # multiplication
    mul = np.array(np.multiply(mat1, mat2))
    # division
    div = np.array(np.divide(mat1, mat2))
    res = (add, sub, mul, div)
    return res
