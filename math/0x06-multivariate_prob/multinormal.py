#!/usr/bin/env python3
"""
    Multivariate Normal distribution
"""
import numpy as np


class MultiNormal:
    """
        Represents a Multivariate Normal distribution
    """
    def __init__(self, data):
        """
        Method:
            Constructor

        Args:
            data[numpy.ndarray], shape (d, n)
               - containing the data set:
                n: the number of data points
                d: the number of dimensions in
                each data point

          Raises:
            - TypeError: If data is not a 2D numpy.ndarray.
            - ValueError: If n is less than 2.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("ata must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data.T, axis=0).reshape(1, d).T
        term = data.T - self.mean.T
        self.cov = np.dot(term.T, term) / n

    def pdf(self, x):
        """
        Method:
            Calculates the PDF at a data point.

        Args:
            x[numpy.ndarray] shape (d, 1)
            containing the data point whose PDF should be calculated

                - d: the number of dimensions of the Multinomial instance

        Returns:
            the value of the PDF
        """
        if not isinstance(x, np.ndarray):
            raise TypeError('x must be a numpy.ndarray')

        dim, _ = self.cov.shape

        if len(x.shape) != 2 or x.shape[1] != 1 or x.shape[0] != dim:
            raise ValueError("x must have the shape ({}, 1)".format(dim))

        mult_term = np.dot(np.dot((x - self.mean).T,
                                  np.linalg.inv(self.cov)),
                           (x - self.mean))

        term1 = (2 * np.pi) ** (dim / 2)

        term2 = np.sqrt(np.linalg.det(self.cov))

        term3 = np.exp((-1 / 2) * mult_term)

        pdf = 1 / (term1 * term2) * term3

        return pdf[0][0]
