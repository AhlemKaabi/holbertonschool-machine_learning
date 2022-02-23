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

    def forward_prop(self, X):
        """
        Method:
            Calculates the forward propagation of the neural network
            Updates the private attributes __A1 and __A2
            The neurons should use a sigmoid activation function.
        Args:
            X (numpy.ndarray): shape (nx, m)
                nx : he number of input features to the neuron
                m : the number of examples
        Returns:
            Returns the private attribute __A1 and __A2
        """
        Z1 = np.matmul(self.__W1, X) + self.b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.matmul(self.__W2, self.__A1) + self.b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Method:
            The cost of the model using logistic regression
        Args:
            Y(numpy.ndarray), shape (1, m):
             That contains the correct labels for the input data
            A(numpy.ndarray), shape(1, m):
              The activated output of the neuron for each
        Returns:
            The cost
        """
        number_examples = Y.shape[1]
        # To avoid division by zero errors,
        # please use 1.0000001 - A instead of 1 - A
        loss = np.matmul(Y,
                         np.log(A).T) + np.matmul((1 - Y),
                                                  np.log(1.0000001 - A).T)
        cost = -np.sum(loss) / number_examples
        return cost

    def evaluate(self, X, Y):
        """
        Method:
            Evaluates the neural network's predictions

        Args:
            X(numpy.ndarray), shape (nx, m):

            - nx is the number of input features to the neuron
            - m is the number of examples

            Y(numpy.ndarra), shape (1, m):
            contains the correct labels for the input data

        Returns:
            the neuron's prediction and the cost of
            the network, respectively
        """
        A1, A2 = self.forward_prop(X)
        # Where True, yield x, otherwise yield y.
        prediction = np.where(A2 >= 0.5, 1, 0)
        cost = self.cost(Y, A2)
        return prediction, cost
