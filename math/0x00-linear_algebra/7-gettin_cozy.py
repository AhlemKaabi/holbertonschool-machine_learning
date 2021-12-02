#!/usr/bin/env python3
"""
    Concatenates two matrices along a specific axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
        Concatenates two matrices along a specific axis.
    """
    # create a copy of mat1
    cp_mat1 = [[columns for columns in rows] for rows in mat1]
    # create a copy of mat2
    cp_mat2 = [[columns for columns in rows] for rows in mat2]

    # concatenate rows
    if axis == 0:
        Res_cat_0 = []
        for rows in cp_mat1:
            Res_cat_0.append(rows)
        for rows in cp_mat2:
            Res_cat_0.append(rows)
        return Res_cat_0
    # concatenate columns
    if axis == 1:
        Res_cat_1 = []
        # matrices must have the same size
        size1 = 0
        size2 = 0
        for _ in cp_mat1:
            size1 += 1
        for _ in cp_mat2:
            size2 += 1
        if size2 != size1:
            return None
        else:
            for index_row in range(size1):
                Res_cat_1.append(cp_mat1[index_row]+cp_mat2[index_row])
            return Res_cat_1
    return None
