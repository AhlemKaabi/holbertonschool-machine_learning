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
    deg = len(poly)
    return [i * poly[i] for i in range(deg) if i - 1 >= 0]
