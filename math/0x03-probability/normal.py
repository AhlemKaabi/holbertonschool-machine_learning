#!/usr/bin/env python3
"""
    class Normal that represents an normal distribution.
"""


class Normal:
    """
        Represents an normal distribution.
    """
    pi = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, mean=0., stddev=1.):
        """
            contructor method

        Args:
            @data: list of the data to be used to
            estimate the distribution.
            @mean: the mean of the distribution.
            @stddev: standard deviation of the distribution.
        """
        # lambda: average time/space == mean!
        if data is None:
            if stddev <= 0:
                raise ValueError("lambtha must be a positive value")
            self.stddev = float(stddev)
            self.mean = float(mean)

        else:
            if isinstance(data, list) is False:
                raise TypeError("data must be a list")
            if(len(data)) < 3:
                raise ValueError("data must contain multiple values")
            # https://en.wikipedia.org/wiki/Standard_deviation
            self.mean = (sum(data) / len(data))
            sum_squre = 0
            for val in data:
                sum_squre += (val - self.mean) ** 2
            variance = sum_squre / len(data)
            self.stddev = variance ** (1/2)

    def z_score(self, x):
        # https://en.wikipedia.org/wiki/Standard_score
        """
            Calculates the z-score of a given x-value

        Args:
            x: the x-value

        Returns:
            the z-score of x
        """

        return((x - self.mean) / self.stddev)

    def x_value(self, z):
        """
            Calculates the x-value of a given z-score

        Args:
            z: the z-score

        Returns:
            the x-value of z
        """
        return(z * self.stddev + self.mean)

    def pdf(self, x):
        """
        Method:
            Calculates the value of the PDF for a given x-value

        Args:
            @x: the x-value

        Returns:
            the PDF value for x
        """
        z = self.z_score(x)
        e = self.e
        pi = self.pi
        stddev = self.stddev
        term1 = e ** -((z ** 2) / 2)
        term2 = stddev * ((2 * pi) ** (1/2))
        pdf = term1 / term2
        return pdf

    def erf(self, x):
        """
            Clacultes the error in a function

        Args:
            x: the x-value

        Returns:
            the error
        """
        pi = self.pi
        term1 = 2 / (pi ** (1/2))
        x1 = x
        x2 = (x ** 3) / 3
        x3 = (x ** 5) / 10
        x4 = (x ** 7) / 42
        x5 = (x ** 9) / 216
        term2 = x1 - x2 + x3 - x4 + x5
        return term1 * term2

    def cdf(self, x):
        """
        Method:
            Calculates the value of the CDF for a given x-value

        Args:
            @x: the x-value

        Returns:
            the CDF value for x
        """

        term = (x - self.mean) / (self.stddev * (2 ** (1/2)))
        cdf = (1/2) * (1 + self.erf(term))
        return cdf
