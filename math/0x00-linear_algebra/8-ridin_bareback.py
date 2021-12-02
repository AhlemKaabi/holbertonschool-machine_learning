#!/usr/bin/env python3
def matrix_shape(matrix):
    """
        Return the shape of a matrix.
    """
    if not type(matrix) == list:
        return []
    return [len(matrix)] + matrix_shape(matrix[0])

def mat_mul(mat1, mat2):
    # check if the two matrices can be multiplied
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)
    if len(shape1) and len(shape2 ) != 2:
        return None
    else:
        if shape1[1] != shape2[0]:
            # cannot be multiplied
            return None
        else:
            result = [[sum(a*b for a,b in zip(mat1_row,mat2_col)) for mat2_col in zip(*mat2)] for mat1_row in mat1]
            return result
            # do multiplication
            # Res_mul =[]
            # Res_rows = []
            # for row_mat1 in mat1:
            #     print("row mat1", row_mat1)
            #     for column_mat2 in range(shape2[1]):
            #         print("this is the size of the columns, ", shape2[1])
            #         print("column index of mat2", column_mat2)
            #         sum = 0
            #         for row_mat2 in range(shape2[0]):
            #             print("test mat2[0][0]",mat2[row_mat2][column_mat2])
            #             break
            #             print("this is the size of the rows, ", shape2[0])
            #             print("row index of mat2", row_mat2)
            #             sum = sum + row_mat1[row_mat2] * mat2[row_mat2][column_mat2]
            #             #print("sum in loop is",sum)
            #         Res_rows.append(sum)
            #         #print("res rows is", Res_rows)
            #         Res_mul.append(Res_rows)
            # return Res_mul
    return None