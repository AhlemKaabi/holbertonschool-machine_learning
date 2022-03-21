#!/usr/bin/env python3
"""
    Correlation
"""
import numpy as np


def correlation(C):
    """
    Method:
        calculates a correlation matrix.

    Args:
        C[numpy.ndarray]:
        - shape (d, d)
        d: number of dimensions

    Raises:
        - TypeError: if C is not a numpy.ndarray
        - ValueError: If C does not have shape (d, d)

    Returns:
        the correlation matrix[numpy.ndarray],
        shape (d, d)
    """

    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if len(C.shape) != 2:
        raise TypeError("C must be a 2D square matrix")
    d1, d2 = C.shape
    if d1 != d2:
        raise ValueError("C must be a 2D square matrix")
    std = np.sqrt(np.diag(C))
    # computationally more efficient according to the formula
    std_x_std = np.outer(std, std)
    return C / std_x_std
