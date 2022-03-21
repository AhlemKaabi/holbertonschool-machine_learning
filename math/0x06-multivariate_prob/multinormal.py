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
        self.cov = np.dot(term.T, term) / n - 1
