#!/usr/bin/env python3
"""
    Calculates the derivative of a polynomial.
"""


def poly_derivative(poly):
    """
        Calculates the derivative of a polynomial.
    """
    if poly == []:
        return [0]
    if type(poly) != list:
        return None
    deg = len(poly)
    if deg == 1:
        return [0]
    return [i * poly[i] for i in range(deg) if i - 1 >= 0]
