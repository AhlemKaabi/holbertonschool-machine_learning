#!/usr/bin/env python3
"""
     Class Neuron that defines a single neuron
     performing binary classification.
"""
import numpy as np


class Neuron:
    """
        Class Neuron.
    """
    def __init__(self, nx):
        """
        Method:
            Constructor
        Args:
            nx : is the number of input features to the neuron
            W : The weights vector for the neuron (random normal
            distribution).
            b : The bias for the neuron.
            A : The activated output of the neuron (prediction)
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        # size=(1, nx) because we have one neuron (one unit)
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    # bundling data with the methods that operate on them.
    # getter for retrieving the data
    # the setter for changing the data. (we don't have stter here!)
    @property
    def W(self):
        """get the weights
        Returns:
            Private instance attribute __W
          """
        return self.__W

    @property
    def b(self):
        """get the bias
        Returns:
            Private instance attribute __b
          """
        return self.__b

    @property
    def A(self):
        """get the activated output
        Returns:
            Private instance attribute __A
          """
        return self.__A

    def forward_prop(self, X):
        """
        Method:
            Calculates the forward propagation of the neuron
		Args:
			X (numpy.ndarray): shape (nx, m)
				nx : he number of input features to the neuron
				m : the number of examples
		Returns:
			Returns the private attribute __A
        """
        z = (self.W * self.A) + self.b
        # new A using sigmoid activation function
        self.__A = 1 / (1 + np.exp(-z))
        return self.A
