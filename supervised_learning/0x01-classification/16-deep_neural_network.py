#!/usr/bin/env python3
"""
    Class DeepNeuralNetwork that defines a deep neural network
     performing binary classification.
"""
import numpy as np


class DeepNeuralNetwork:
    """
        DeepNeuralNetwork
    """
    def __init__(self, nx, layers):
        """
        Method:
            Constructor
        Args:
            nx : the number of input features to the neuron.
            layers(list): the number of nodes in each layer
            of the network

        * Public instance attributes *

        L: The number of layers in the neural network.

        cache(dictionary): to hold all intermediary values
        of the network. Upon instantiation, it should be set
        to an empty dictionary.

        weights(dictionary): to hold all weights and biased
        of the network
        """
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) != list:
            raise TypeError('layers must be a list of positive integers')
        if all(i < 0 for i in layers):
            raise TypeError('layers must be a list of positive integers')
        if len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')

        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        # weights initializatio --> to read!!!
        # https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
        # https://towardsdatascience.com/random-initialization-for-neural-networks-a-thing-of-the-past-bfcdd806bf9e
        # he et al:
        # the weights are initialized keeping in mind the size of the previous
        # layer which helps in attaining a global minimum of the cost function
        # faster and more efficiently.
        for l_ in range(self.L):
            if l_ == 0:
                rand = np.random.randn(layers[l_], nx)
                sqrt = np.sqrt(2 / nx)
                self.weights['W' + str(l_ + 1)] = rand * sqrt
            else:
                rand = np.random.randn(layers[l_], layers[l_ - 1])
                sqrt = np.sqrt(2 / layers[l_ - 1])
                self.weights['W' + str(l_ + 1)] = rand * sqrt
            self.weights['b' + str(l_)] = np.zeros((layers[l_], 1))
