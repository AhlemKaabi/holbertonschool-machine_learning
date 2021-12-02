#!/usr/bin/env python3
"""
	Calculates the shape of a matrix.
"""
def matrix_shape(matrix):
	"""
		Return the shape of a matrix.
	"""
	if not type(matrix) == list:
		return []
	return [len(matrix)] + matrix_shape(matrix[0])
