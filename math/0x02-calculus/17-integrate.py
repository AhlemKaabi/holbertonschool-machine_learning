#!/usr/bin/env python3


"""
    Calculates the integral of a polynomial.
"""


def poly_integral(poly, C=0):
    """
        Calculates the integral of a polynomial.
    """
    integral = [0]

    for count, value in enumerate(poly):
        integral.append(value * (1 / (count + 1)))
    return integral
