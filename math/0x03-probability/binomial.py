#!/usr/bin/env python3
"""
    class Binomial that represents a binomial distribution.
"""


class Binomial:
    """ represents a binomial distribution. """
    def __init__(self, data=None, n=1, p=0.5):
        """
            Contructor method
        Args:
              @data: a list of the data to be used to estimate the distribution
            @n: the number of Bernoulli trials
            @p: the probability of a “success”
           """
        self.n = int(n)
        self.p = float(p)
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if (p <= 0) or (p >= 1):
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if(len(data)) < 3:
                raise ValueError("data must contain multiple values")
            mean = (sum(data) / len(data))
            sum_square = 0
            for val in data:
                sum_square += (val - mean) ** 2
            variance = sum_square / len(data)
            p = 1 - (variance / mean)
            self.n = round(mean / p)
            self.p = mean / self.n

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
            return n * self.fact(n - 1)

    def pmf(self, k):
        """
            PMF
        """
        if isinstance(k, int) is False:
            k = int(k)
        if k < 0:
            return 0
        term1 = self.fact(self.n)
        term2 = (self.fact(k) * self.fact(self.n - k))
        binomial_coef = term1 / term2
        term3 = (self.p ** k) * ((1 - self.p) ** (self.n - k))
        pmf = binomial_coef * term3
        return pmf

    def cdf(self, k):
        """
            CDF
        """
        k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return (cdf)
