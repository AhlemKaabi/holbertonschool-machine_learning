#!/usr/bin/env python3
"""
    Add two matrices.
"""


def matrix_shape(matrix):
    """
        Return the shape of a matrix.
    """
    if not type(matrix) == list:
        return []
    return [len(matrix)] + matrix_shape(matrix[0])


def add_matrices(mat1, mat2):
    """
        Add two matrices.
    """

    if matrix_shape(mat1) != matrix_shape(mat2):
        return None

    add = mat1.copy()

    print(add)
    for i , j in zip(range(len(mat1)), range(len(mat2))):
        for k, h in zip(range(i), range(j)):
            add[i][k] = mat1[i][k] + mat2[j][h]