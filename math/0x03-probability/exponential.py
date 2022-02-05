#!/usr/bin/env python3
"""
    class Exponential that represents an exponential distribution.
"""


class Exponential:
    """
        Represents an exponential distribution.
    """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
            contructor method

        Args:
            @data: list of the data to be used to
               estimate the distribution
            @lambtha: the expected number of occurences in a
               given time frame
        """
        # lambda: average time/space
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if (type(data) is not list):
                raise TypeError("data must be a list")
            if (len(data) < 2):
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            self.lambtha = float(1 / mean)

    def pdf(self, x):
        """
        Method:
            Calculates the value of the PDF for a given time period

        Args:
            @x: the time period

        Returns:
            the PDF value for x
        """
        if x < 0:
            return 0
        return self.lambtha * (self.e ** (-self.lambtha * x))

    def cdf(self, x):
        """
        Method:
            Calculates the value of the CDF for a given time period

        Args:
            @x: the time period

        Returns:
            the PDF value for x
        """
        if x < 0:
            return 0

        term1 = (self.e ** (-self.lambtha * x))
        return 1 - term1
