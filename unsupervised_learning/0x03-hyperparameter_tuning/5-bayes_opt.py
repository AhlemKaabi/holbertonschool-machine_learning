#!/usr/bin/env python3
"""
    Bayesian Optimization
"""
import numpy as np
from scipy.stats import norm
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

    def acquisition(self):
        """
        Method to calculate the next best sample location.
        ** Uses the Expected Improvement acquisition function **

        Returns: X_next, EI
            X_next (numpy.ndarray of shape (1,)):
             representing the next best sample point.

            EI (numpy.ndarray of shape (ac_samples,)):
             containing the expected improvement of each potential sample.
        """
        # https://colab.research.google.com/github/krasserm/bayesian-machine-learning/blob/dev/bayesian-optimization/bayesian_optimization.ipynb#scrollTo=4lsMKUsR4w5m
        mu, sigma = self.gp.predict(self.X_s)

        # Z = np.zeros(sigma.shape[0])

        # optimization
        if (self.minimize):
            # the function f: value of the best sample
            mu_sample_opt = np.min(self.gp.Y)
            # minimize => reduce the mean!
            improve = mu_sample_opt - mu - self.xsi
        else:
            # the function f: value of the best sample
            mu_sample_opt = np.max(self.gp.Y)
            improve = mu - mu_sample_opt - self.xsi

        # for i in range(sigma.shape[0]):
        #     if sigma[i] > 0:
        #         Z[i] = improve[i] / sigma[i]
        #     else:
        #         Z[i] = 0
        #     EI = improve * norm.cdf(Z) + sigma * norm.pdf(Z)
        Z = np.where(sigma == 0, 0, improve / sigma)
        ei = improve * norm.cdf(Z) + sigma * norm.pdf(Z)
        EI = np.where(sigma == 0, 0, ei)
        # EI = np.maximum(EI, 0)
        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """
        Method to optimize the black-box function.

        Parameters:
            iterations: the maximum number of iterations to perform.

        Returns: X_opt, Y_opt
            X_opt (numpy.ndarray of shape (1,)):
             representing the optimal point.
            Y_opt (numpy.ndarray of shape (1,)):
             representing the optimal function value.
        """
        # If the next proposed point is one that has already been sampled,
        # optimization should be stopped early

        for _ in range(iterations):
            # next best sample location
            X_next, _ = self.acquisition()

            if X_next in self.gp.X:
                break
            # next best function value of the sample location
            Y_next = self.f(X_next)
            # update
            self.gp.update(X_next, Y_next)

        # idx_opt_f: index of the optimized (max or min) black-box function
        if self.minimize:
            idx_opt_f = np.argmin(self.gp.Y)
        else:
            idx_opt_f = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx_opt_f]
        Y_opt = self.gp.Y[idx_opt_f]
        self.gp.X = self.gp.X[:-1]
        return X_opt, Y_opt
