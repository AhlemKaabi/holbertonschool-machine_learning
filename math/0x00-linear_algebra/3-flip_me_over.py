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
    size = matrix_shape(matrix)
    rows = size[1]
    colums = size[0]
    transpose_matrix = [[0] * colums for i in range(rows)]
    # return numpy.transpose(matrix).array
    transpose_matrix_rows = 0
    for row_matrix in matrix:
        transpose_matrix_colums = 0
        for column_matrix_elem in row_matrix:
            transpose_matrix[transpose_matrix_colums][transpose_matrix_rows] = column_matrix_elem
            transpose_matrix_colums += 1
        transpose_matrix_rows += 1
    return transpose_matrix