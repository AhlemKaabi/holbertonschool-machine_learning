#!/usr/bin/env python3
def matrix_shape(matrix):
    """
        Return the shape of a matrix.
    """
    if not type(matrix) == list:
        return []
    return [len(matrix)] + matrix_shape(matrix[0])


def add_matrices2D(mat1, mat2):

    shape_mat1 = matrix_shape(mat1)
    shape_mat2 = matrix_shape(mat2)

    if shape_mat1 != shape_mat2:
        return None


    if len(shape_mat1) == 2:
        sum =[ [0] * shape_mat1[1] for i in range(shape_mat1[0])]

    for idx_rows in range(shape_mat1[0]):
        for idx_columns in range(shape_mat1[1]):
            sum[idx_rows][idx_columns] = mat1[idx_rows][idx_columns] + mat2[idx_rows][idx_columns]
    return sum