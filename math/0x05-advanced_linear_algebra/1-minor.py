#!/usr/bin/env python3
""" Minor matrix of a matrix in pure python"""


def check_squareness(A):
    """
    Makes sure that a matrix is square
        :param A: The matrix to be checked.
    """
    if len(A) != len(A[0]):
        raise ValueError("matrix must be a non-empty square matrix")


def zeros_matrix(rows, cols):
    """
    Creates a matrix filled with zeros.
        :param rows: the number of rows the matrix should have
        :param cols: the number of columns the matrix should have
        :return: list of lists that form the matrix
    """
    M = []
    while len(M) < rows:
        M.append([])
        while len(M[-1]) < cols:
            M[-1].append(0.0)

    return M


def copy_matrix(M):
    """
    Creates and returns a copy of a matrix.
        :param M: The matrix to be copied
        :return: A copy of the given matrix
    """
    # Section 1: Get matrix dimensions
    rows = len(M)
    cols = len(M[0])

    # Section 2: Create a new matrix of zeros
    MC = zeros_matrix(rows, cols)

    # Section 3: Copy values of M into the copy
    for i in range(rows):
        for j in range(cols):
            MC[i][j] = M[i][j]

    return MC


def determinant(matrix):
    """
    Method:
        Calculates the determinant of a matrix

     Args:
        matrix: (list of lists) whose
          determinant should be calculated

     Returns:
         The determinant of matrix
    """
    # https://integratedmlai.com/find-the-determinant-of-a-matrix-with-pure-python-without-numpy-or-scipy/

    if len(matrix) == 1:
        return matrix[0][0]
    total = 0

    A = matrix
    # Section 1: store indices in list for flexible row referencing
    indices = list(range(len(A)))

    # Section 2: when at 2x2 submatrices recursive calls end
    if len(A) == 2 and len(A[0]) == 2:
        val = A[0][0] * A[1][1] - A[1][0] * A[0][1]
        return val

    # Section 3: define submatrix for focus column and call this function
    for fc in indices:  # for each focus column, find the submatrix ...
        As = copy_matrix(A)  # make a copy, and ...
        As = As[1:]  # ... remove the first row

        height = len(As)

        for i in range(height):  # for each remaining row of submatrix ...
            As[i] = As[i][0:fc] + As[i][fc+1:]  # zero focus column elements

        sign = (-1) ** (fc % 2)  # alternate signs for submatrix multiplier
        sub_det = determinant(As)  # pass submatrix recursively
        total += sign * A[0][fc] * sub_det  # total all returns from recursion

    return total


def minor(matrix):
    """
    Method:
        Calculates the minor matrix of a matrix

     Args:
        matrix: (list of lists) whose
        minor matrix should be calculated

     Returns:
         The minor matrix of matrix
    """
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    mat_len = len(matrix)
    if mat_len == 0:
        raise TypeError("matrix must be a list of lists")
    if mat_len == 1:
        return [[1]]
    # for i in range(mat_len):
    #     if not isinstance(matrix[i], list):
    #         raise TypeError("matrix must be a list of lists")

    # if matrix == [[]]:
    #     return 1

    for i in range(len(matrix)):
        if not isinstance(matrix[i], list) or not len(matrix[i]):
            raise TypeError("matrix must be a list of lists")

        if len(matrix) != len(matrix[i]):
            raise ValueError("matrix must be a non-empty square matrix")

    minor = []
    for i in range(mat_len):
        row = []
        for j in range(mat_len):
            mat = [vec[:] for vec in matrix]
            del mat[i]
            for line in mat:
                del line[j]
            det = determinant(mat)
            row.append(det)

        # append to the minor
        minor.append(row)

    return minor
