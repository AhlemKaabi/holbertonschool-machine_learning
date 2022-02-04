#!/usr/bin/env python3
"""
    Class Poisson that represents a poisson distribution.
"""


class Poisson:
    """
        Class Poisson that represents a poisson distribution.
    """

    e = 2.7182818285

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
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """
        Method:
            Calculates the value of the PMF for
            a given number of “successes”

        Args:
            @k: the number of “successes”

        Returns:
            the PMF value for k
        """
        def fact(n):
            """
                calculates factorial of given number
            """
            if n == 1 or n == 0:
                return 1
            else:
                return n * fact(n-1)
        k = int(k)
        if k < 0:
            return 0
        return (self.lambtha ** k * self.e ** -self.lambtha) / fact(k)

    def cdf(self, k):
        """"
        Method:
            Calculates the value of the CDF for
            a given number of “successes”

        Args:
            @k: the number of “successes”

        Returns:
            the PMF value for k
        """
        k = int(k)
        if k <= 0:
            return 0

        cdf = 0
        for k in range(1, k + 1):
            cdf += self.pmf(k)
        return
