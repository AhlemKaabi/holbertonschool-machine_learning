#!/usr/bin/env python3
"""
    Definiteness
"""
import numpy as np


def definiteness(matrix):
    """
    Method:
        Calculates the definiteness of a matrix.

     Args:
         matrix(numpy.ndarray), shape (n, n):
           whose definiteness should be calculated

     Returns:
        - Positive definite
        - Positive semi-definite
        - Negative semi-definite
        - Negative definite
        - Indefinite
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) == 1:
        return None
    row, col = matrix.shape
    if row != col:
        return None
    # not a valid matrix (not symetric)
    if not np.allclose(matrix, matrix.T):
        return None

    w, _ = np.linalg.eig(matrix)
    semi_def = 0
    if 0 in w:
        semi_def = 1
    # positive definite
    if all([elem >= 0 for elem in w]) and not semi_def:
        return "Positive definite"
    # negartive definit
    elif all([elem <= 0 for elem in w]) and not semi_def:
        return "Negative definite"
    # positive semidefinit
    elif all([elem >= 0 for elem in w]) and semi_def:
        return "Positive semi-definite"
    # negative semidefinit
    elif all([elem <= 0 for elem in w]) and semi_def:
        return "Negative semi-definite"
    # indefinite
    else:
        return "Indefinite"
