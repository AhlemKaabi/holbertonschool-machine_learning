#!/usr/bin/env python3

if __name__ == '__main__':
    cofactor = __import__('2-cofactor').cofactor

    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]

    print(cofactor(mat1))
    try:
        cofactor(mat1)
    except Exception as e:
        print(e)
    print(cofactor(mat2))
    print(cofactor(mat3))
    print(cofactor(mat4))
    try:
        cofactor(mat5)
    except Exception as e:
        print(e)
    try:
        cofactor(mat6)
    except Exception as e:
        print(e)

    # try:
    #     cofactor([[]])
    # except ValueError as e:
    #     print(str(e))
    # try:
    #     cofactor([[1], [1]])
    # except ValueError as e:
    #     print(str(e))
    # try:
    #     cofactor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    # except ValueError as e:
    #     print(str(e))
    # try:
    #     cofactor([[1, 2, 3], [1, 2, 3, 4], [1, 2, 3]])
    # except ValueError as e:
    #     print(str(e))
