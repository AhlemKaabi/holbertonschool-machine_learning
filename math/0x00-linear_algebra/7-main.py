#!/usr/bin/env python3

# cat_matrices2D = __import__('7-gettin_cozy').cat_matrices2D

# mat1 = [[1, 2], [3, 4]]
# mat2 = [[5, 6]]
# mat3 = [[7], [8]]
# mat4 = cat_matrices2D(mat1, mat2)
# mat5 = cat_matrices2D(mat1, mat3, axis=1)
# print(mat4)
# print(mat5)
# mat1[0] = [9, 10]
# mat1[1].append(5)
# print(mat1)
# print(mat4)
# print(mat5)

# import sys
# cat_matrices2D = __import__('7-gettin_cozy').cat_matrices2D
# m1 = [[4, -7, 56, 2], [5, 106, 7, 2]]
# m2 = [[2, -6, 3], [0, -6, 3]]
# m = cat_matrices2D(m1, m2)
# print(m)
# m1 = [[484, 247], [554, 16], [5, 88]]
# m2 = [[233, -644, 325], [406, -16, 33], [765, 34, -39]]
# m = cat_matrices2D(m1, m2, axis=0)
# print(m)
# m1 = [[-54, -87, 95], [54, 16, -72]]
# m2 = [[12, 63, 79], [-10, 69, -9], [76, 45, -11]]
# m = cat_matrices2D(m1, m2, axis=1)
# print(m)


import sys
cat_matrices2D = __import__('7-gettin_cozy').cat_matrices2D
m1 = [[51, 24, 73], [93, 45, 77]]
m2 = [[], [], []]
m = cat_matrices2D(m1, m2)
print(m)
m2 = [[51, 24, 73], [93, 45, 77]]
m1 = [[], [], []]
m = cat_matrices2D(m1, m2, axis=0)
print(m)
m1 = [[75, 23, 58], [32, 5, 67], [34, 65, 22]]
m2 = [[], [], []]
m = cat_matrices2D(m1, m2, axis=1)
if type(m) is not list or m is m1 or m is m2 or not len(m) or type(m[0]) is not list:
    print("Not a new matrix")
    sys.exit(1)
print(m)