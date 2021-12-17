#!/usr/bin/env python3

"""
    Class Poisson that represents a poisson distribution:
"""

class Poisson:
    def __init__(self, data=None, lambtha=1.):
        """
            Args:
                data: list of the data to be used to estimate the distribution.
                lambtha: the expected number of occurences in a given time frame.
            Raises:
                ValueError: If size is less than 0
        """
        self.lambtha = float(lambtha)
        if not data :
            if (lambtha <= 0):
                raise ValueError("lambtha must be a positive value")
        else:
            if (type(data) is not list):
                raise TypeError("data must be a list")
            if (len(data) < 2):
                 raise ValueError("data must contain multiple values")
