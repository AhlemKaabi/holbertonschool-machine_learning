#!/usr/bin/env python3
"""
 class Neuron that defines a single neuron
 performing binary classification.
"""
import numpy as np

class Neuron:
    """
        class Neuron.
    """
    def __init__(self, nx):
        """
            Args:
                nx : is the number of input features to the neuron
                W : The weights vector for the neuron (random normal distribution).
                b : The bias for the neuron.
                A : The activated output of the neuron (prediction)
        """
        if type(nx) is not int:
            raise TypeError("nx must be n integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__nx = nx
        self.__W = np.random.normal(size=(1, 784))
        self.__b = 0
        self.__A = 0

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
