#!/usr/bin/env python3


"""
    Calculates the integral of a polynomial.
"""


def poly_integral(poly, C=0):
    """
        Calculates the integral of a polynomial.
    """
    if poly == [] or C == None or type(poly) == int:
        return None

    if poly is [0]:
        return [C]

    integral = [C]
# https://python-reference.readthedocs.io/en/latest/docs/float/index.html
    for count, value in enumerate(poly):
        res = value / (count + 1)
        if res.is_integer():
            integral.append(int(res))
        else:
            integral.append(res)
    return integral
