#!/usr/bin/env python3
"""
    Gaussian Process
"""
import numpy as np


class GaussianProcess:
    """
        Class that represents a noiseless 1D Gaussian process.
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
            Class constructor

            Parameters:
                - X_init (numpy.ndarray of shape (t, 1)):
                representing the inputs already sampled with the black-box
                function

                - Y_init (numpy.ndarray of shape (t, 1)):
                representing the outputs of the black-box function for each
                input in X_init.
                    - t (ijnt): the number of initial samples.

                - l (int): the length parameter for the kernel

                - sigma_f (float): the standard deviation given to the output
                of the black-box function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        # the properties of a kernel (i.e. semi-positive definite and
        # symmetric)
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
            Method to compute a covariance kernel matrix between two matrices.
            Isotropic squared exponential kernel (same length parameter l for
            all input dimensions)

            Parameters:

                X1 (numpy.ndarray of shape (m, 1))
                X2 (numpy.ndarray of shape (n, 1))

            Returns:
                the covariance kernel matrix as a numpy.ndarray of shape (m, n)

            ** the kernel should use the Radial Basis Function (RBF) **
        """
        # This kernel has two hyperparameters:
        # 	- signal variance, σ²: the vertical variation (self.sigma_f)
        # 	- lengthscale, l: controls the smoothness of the function (self.l)

        x1 = np.sum(X1 ** 2, 1).reshape(-1, 1)
        x2 = np.sum(X2 ** 2, 1)
        dot = 2 * np.dot(X1, X2.T)
        sqdist = x1 + x2 - dot

        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)
