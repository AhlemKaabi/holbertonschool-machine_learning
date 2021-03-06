#!/usr/bin/env python3
"""
    Class NeuralNetwork that defines a neural network with
    one hidden layer performing binary classification.
"""
import numpy as np


class NeuralNetwork:
    """
    Class NeuralNetwork
    """
    def __init__(self, nx, nodes):
        """
        Method:
            Constructor
        Args:
            nx : the number of input features to the neuron.
            nodes: the number of nodes found in the hidden layer.


        * Public instance attributes *

        W1: The weights vector for the hidden layer.
        b1: The bias for the hidden layer.
        A1: The activated output for the hidden layer.
        W2: The weights vector for the output neuron.
        b2: The bias for the output neuron.
        A2: The activated output for the output neuron (prediction).
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b1 = np.zeros((nodes, 1))
        self.__b2 = 0
        self.__A1 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """get the weights
        Returns:
            Private instance attribute __W1
          """
        return self.__W1

    @property
    def b1(self):
        """get the bias
        Returns:
            Private instance attribute __b1
          """
        return self.__b1

    @property
    def A1(self):
        """get the activated output
        Returns:
            Private instance attribute __A1
          """
        return self.__A1

    @property
    def W2(self):
        """get the weights
        Returns:
            Private instance attribute __W2
          """
        return self.__W2

    @property
    def b2(self):
        """get the bias
        Returns:
            Private instance attribute __b2
          """
        return self.__b2

    @property
    def A2(self):
        """get the activated output
        Returns:
            Private instance attribute __A2
          """
        return self.__A2
