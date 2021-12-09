#!/usr/bin/env python3
"""
    Silce a numpy array with Dictionary of axes as keys
    and slices as values
"""


def np_slice(matrix, axes={}):
    """
        matrix: is the matrix to slice
        axes={axis number: slice,...}
    """
    # to slice a np.array we need a tuple of slices!
    # matrix[tuple(slices)]
    # slices should be ordered, first axis, second ...
    mat_slices = []

    for axis_num in range(matrix.ndim):
        # get the right slice for axis_num
        tuple_slice = axes.get(axis_num)
        if tuple_slice is not None:
            mat_slices.append(slice(*tuple_slice))
        else:
            mat_slices.append(slice(None))

    return matrix[tuple(mat_slices)]
