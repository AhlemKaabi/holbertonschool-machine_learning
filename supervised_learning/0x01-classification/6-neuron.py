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

        * The neuron should use a sigmoid activation function *
        """
        # print("shape of X", X.shape) -> (784, 12665)
        # print("shape of __W", self.__W.shape) -> (1, 784)
        # print("value of __b", self.__b) -> 0 (1, 1)
        Z = np.matmul(self.__W, X) + self.b
        # print("shape of Z", Z.shape)
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Method:
            Calculates the cost of the model using
            logistic regression.
        Args:
            - Y (numpy.ndarray) : shape (1, m) that contains the
            correct labels for the input data.
            - A (numpy.ndarray) : shape (1, m) containing the
            activated output of the neuron for each example.
        """
        number_examples = Y.shape[1]
        loss = np.matmul(Y,
                         np.log(A).T) + np.matmul((1 - Y),
                                                  np.log(1.0000001 - A).T)
        cost = -np.sum(loss) / number_examples
        return cost

    def evaluate(self, X, Y):
        """
        Method:
            Evaluates the neuron's predictions

        Args:
            X(numpy.ndarray), shape (nx, m): contains the input data
             - nx: number of input features to the neuron
             - m: number of examples

            Y (numpy.ndarray), shape (1, m):
            contains the correct labels for the input data

        Return:
            the neuron's prediction and the cost of the network
             - The prediction: (numpy.ndarray), shape (1, m):
                containing the predicted labels for each example
             - The label values: 1 if the output of the network is >= 0.5
                and 0 otherwise
        """
        A = self.forward_prop(X)
        # Where True, yield x, otherwise yield y.
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Method:
            Calculates one pass of gradient descent on the neuron
        Args:
            X(numpy.ndarray), shape (nx, m): contains the input data
             - nx is the number of input features to the neuron
             - m is the number of examples
            Y(numpy.ndarray), shape (1, m): contains the correct
             labels for the input data
            A(numpy.ndarray), shape (1, m): containing the activated
             output of the neuron for each example
            alpha: learning rate

        * Updates the private attributes __W and __b *
        """
        number_examples = X.shape[1]
        # print("shape of X", X.shape) -> (784, 12665)
        # print("shape of Y", Y.shape)
        dZ = A - Y
        # print("shape of dZ", dZ.shape) -> (1, 12665)
        dW = np.matmul(X, dZ.T)
        dW /= number_examples
        # print("shape of dW", dW.shape) -> (784, 1)
        db = np.sum(dZ)
        db /= number_examples
        # print("shape of db", db.shape) -> ()
        self.__W -= alpha * dW.T
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Method:
            Trains the neuron

        Args:
            X(numpy.ndarray), shape (nx, m): contains the input data
             - nx is the number of input features to the neuron
             - m is the number of examples

            Y(numpy.ndarray), shape (1, m): contains the correct
             labels for the input data

            iterations: number of iterations to train over

            alpha: the learning rate

        Returns:
            evaluation of the training data after iterations
            of training have occurred
        """
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")

        if iterations < 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) != float:
            raise TypeError("alpha must be a float")

        if alpha < 0:
            raise ValueError("alpha must be positive")

        for i in iterations:
            A = self.forward_prop()
            self.gradient_descent(X, Y, A)

        prediction, cost = self.evaluate(X, Y)

        return prediction, cost
