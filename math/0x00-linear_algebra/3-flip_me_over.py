#!/usr/bin/env python3
"""
    Transpose of a 2D matrix.
"""


def matrix_shape(matrix):
    """
        Return the shape of a matrix.
    """
    if not type(matrix) == list:
        return []
    return [len(matrix)] + matrix_shape(matrix[0])


def matrix_transpose(matrix):
    """
        Returns the transpose of a 2D matrix.
    """
    size = matrix_shape(matrix)
    rows = size[1]
    colums = size[0]
    tran_matrix = [[0] * colums for i in range(rows)]
    tran_matrix_ro = 0
    for row_matrix in matrix:
        tran_matrix_col = 0
        for column_matrix_elem in row_matrix:
            tran_matrix[tran_matrix_col][tran_matrix_ro] = column_matrix_elem
            tran_matrix_col += 1
        tran_matrix_ro += 1
    return tran_matrix
