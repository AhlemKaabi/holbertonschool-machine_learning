#!/usr/bin/env python3

"""
    that calculates Sum Sigma
"""


def summation_i_squared(n):
    """
        that calculates Sum Sigma
    """

    if n >= 0:
        res = (n * (n + 1) * ((2 * n) + 1))/6
        return int(res)
    return None
