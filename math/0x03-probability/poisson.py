#!/usr/bin/env python3

"""
    Class Poisson that represents a poisson distribution.
"""


class Poisson:
    """
    Class Poisson that represents a poisson distribution.
    """
    def __init__(self, data=None, lambtha=1.):
        """
            Args:
                data: list of the data to be used to estimate
                the distribution.
                lambtha: the expected number of occurences
                in a given time frame.

            Raises:
                ValueError: lambtha must be a positive value
                TypeError: data must be a list
                ValueError: data must contain multiple values
        """
        self.lambtha = float(lambtha)
        if data is None:
            if (lambtha <= 0):
                raise ValueError("lambtha must be a positive value")
        else:
            if (type(data) is not list):
                raise TypeError("data must be a list")
            if (len(data) < 2):
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))
