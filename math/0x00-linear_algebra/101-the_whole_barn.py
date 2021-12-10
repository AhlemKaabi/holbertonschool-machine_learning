#!/usr/bin/env python3
"""
    Add two matrices.
"""
import itertools


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
    # access all combinations at the same time
    # say that we have a matrices of shape[3, 2, 2]
    # combination of
    #              /  (0, 0, 0)
    #              |  (0, 0, 1)
    #              |  (0, 1, 0)
    # 0   0   0    |  (0, 1, 1)
    # 1   1   1    |  (1, 0, 0)
    # 2           <   (1, 0, 1)
    #              |  (1, 1, 0)
    #              |  (1, 1, 1)
    #              |  (2, 0, 0)
    #              |  (2, 0, 1)
    #              |  (2, 1, 0)
    #              \  (2, 1, 1)
    # all these combination is how we access the matrix

    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    shape = matrix_shape(mat1)
    # sum_matrix is just a copy of any matrix to get the right shape
    sum_matrix = mat1.copy()
    # elements_to_combine is a tuple of [0, 1, 2] , [0,1] [0,1]
    elements_to_combine = tuple(range(i) for i in shape)
    # combination_list is a list that comtains all the
    # combination to access any of our matrices
    combination_list = [i for i in itertools.product(*elements_to_combine)]

    for X in combination_list:
        # mat_elem_acc is the map to our elemnt [0][0][0] !
        mat_elem_acc = '['
        for idx, tup_elem in enumerate(X):
            if idx >= 0 and idx < len(X) - 1:
                mat_elem_acc += str(tup_elem) + ']['
            else:
                mat_elem_acc += str(tup_elem) + ']'
        exec('sum_matrix' + mat_elem_acc + '=' +
             'mat1' + mat_elem_acc + '+' +
             'mat2' + mat_elem_acc)

    return sum_matrix
