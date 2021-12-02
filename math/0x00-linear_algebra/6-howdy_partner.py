#!/usr/bin/env python3
"""
    Concatenates two arrays.
"""


def cat_arrays(arr1, arr2):
    """
        Concatenates two arrays.
    """
    cat_arr = []

    for elem1 in arr1:
        cat_arr.append(elem1)

    for elem2 in arr2:
        cat_arr.append(elem2)

    return cat_arr
