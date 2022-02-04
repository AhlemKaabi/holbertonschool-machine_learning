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
        self.pi = 3.1415926536
        self.e = 2.7182818285

        if data is None:
            if (lambtha <= 0):
                raise ValueError("lambtha must be a positive value")
        else:
            if (type(data) is not list):
                raise TypeError("data must be a list")
            if (len(data) < 2):
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def fact(self, n):
        """
        Method:
            calculates factorial of given number

        Args:
            @n: number

        Returns:
            factorial of given number
        """

        if n == 1 or n == 0:
            return 1
        else:
            return  n * self.fact(n - 1);

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
        if type(k) != int:
            k = int(k)

        if k > 0:
            p = ((self.e ** -self.lambtha) * (self.lambtha ** k)) / self.fact(k)
            return p
        else:
            return 0