#!/usr/bin/env python3
"""
    Bayesian Optimization
"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
        Performs Bayesian optimization on a noiseless 1D Gaussian process.
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):

        """
        Class constructor

        Parameters:
            f: the black-box function to be optimized

            X_init (numpy.ndarray of shape (t, 1)):
                representing the inputs already sampled with the black-box
                function

            Y_init (numpy.ndarray of shape (t, 1))
                representing the outputs of the black-box function for each
                input in X_init

            t (int): the number of initial samples

            bounds (tuple of (min, max)):
                representing the bounds of the space in which to look for the
                optimal point

            ac_samples (int): the number of samples that should be analyzed
            during acquisition

            l (int): the length parameter for the kernel

            sigma_f: the standard deviation given to the output of the
            black-box function

            xsi: the exploration-exploitation factor for acquisition

            minimize (bool): determining whether optimization should be
            performed for minimization (True) or maximization (False)
        """

        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        # self.X_s: (numpy.ndarray of shape (ac_samples, 1)): containing
        # all acquisition sample points, evenly spaced between min and max
        # https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
        # np.linspace; Return evenly spaced numbers over a specified interval.
        min_, max_ = bounds
        self.X_s = np.linspace(min_, max_, ac_samples).reshape(ac_samples, 1)
        self.xsi = xsi
        self.minimize = minimize
