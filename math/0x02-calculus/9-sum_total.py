#!/usr/bin/env python3

"""
    that calculates Sum Sigma
"""


def summation_i_squared(n):
    """
        that calculates Sum Sigma
    """

    if n < 0 or type(n) is not int:
        return None
    res = (n * (n + 1) * ((2 * n) + 1))/6
    return res
