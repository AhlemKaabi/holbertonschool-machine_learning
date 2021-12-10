#!/usr/bin/env python3
"""
    Adds two matrices element-wise.
"""


def matrix_shape(matrix):
    """
        Return the shape of a matrix.
    """
    if not type(matrix) == list:
        return []
    if matrix == []:
        return [0]
    return [len(matrix)] + matrix_shape(matrix[0])


def add_matrices2D(mat1, mat2):
    """
        Adds two matrices element-wise.
    """
    # make sure that the matrices are not empty
    if len(mat1) == 0 or len(mat2) == 0:
        return None

    shape_mat1 = matrix_shape(mat1)
    shape_mat2 = matrix_shape(mat2)
    if shape_mat1 != shape_mat2:
        return None

    if len(shape_mat1) == 2:
        sum = [[0] * shape_mat1[1] for i in range(shape_mat1[0])]

    for i_rows in range(shape_mat1[0]):
        for i_col in range(shape_mat1[1]):
            sum[i_rows][i_col] = mat1[i_rows][i_col] + mat2[i_rows][i_col]

    return sum
