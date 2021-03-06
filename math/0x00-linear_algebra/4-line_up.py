#!/usr/bin/env python3
"""
    Adds two arrays element-wise
"""


def add_arrays(arr1, arr2):
    """
        Adds two arrays element-wise
    """
    size_arr1 = len(arr1)
    size_arr2 = len(arr2)

    if size_arr1 != size_arr2:
        return None

    sum = []
    for idx in range(size_arr1):
        sum.append(arr1[idx] + arr2[idx])
    return sum
