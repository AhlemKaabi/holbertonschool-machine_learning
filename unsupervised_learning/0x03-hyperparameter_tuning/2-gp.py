#!/usr/bin/env python3
"""
    Gaussian Process

    Machine learning - Introduction to Gaussian processes:
        https://www.youtube.com/watch?v=4vGiHC35j9s


    Notes:
    - dot product mesure similarity
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

    def predict(self, X_s):
        """
        Method to predict the mean and standard deviation of points in
        a Gaussian process.

        Parameters:
            X_s (numpy.ndarray of shape (s, 1)):
             containing all of the points whose mean and standard deviation
            should be calculated.
                - s: the number of sample points

        Returns: mu, sigma
            mu (numpy.ndarray of shape (s,)):
            containing the mean for each point in X_s, respectively.
            sigma (numpy.ndarray of shape (s,)):
            containing the variance for each point in X_s, respectively.
        """
        # https://colab.research.google.com/drive/1PvmTJWWRy4e5CaI93m7i49AY6xIMTlnY#scrollTo=GpWG-ENMrxoy&line=1&uniqifier=1
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        # calculate mu
        mu = K_s.T.dot(K_inv).dot(self.Y).reshape((X_s.shape[0]))

        # calculate cov
        cov = K_ss - K_s.T.dot(K_inv).dot(K_s)

        sigma = np.diag(cov)

        return mu, sigma

    def update(self, X_new, Y_new):
        """
        Method to update a Gaussian Process.

        Parameters:
            X_new (numpy.ndarray of shape (1,)):
             that represents the new sample point.
            Y_new (numpy.ndarray of shape (1,)):
             that represents the new sample function value.

        * Updates the public instance attributes X, Y, and K *
        """
        # self.X = np.append(self.X, X_new).reshape(-1, 1)
        # self.Y = np.append(self.Y, Y_new).reshape(-1, 1)
        # Stack arrays in sequence vertically (row wise).
        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))
        self.K = self.kernel(self.X, self.X)
