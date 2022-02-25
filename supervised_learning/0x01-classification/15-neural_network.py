#!/usr/bin/env python3
"""
    Class NeuralNetwork that defines a neural network with
    one hidden layer performing binary classification.
"""
import numpy as np
import matplotlib.pyplot as plt



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

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Method:
            Calculates *one pass* of gradient descent on
            the neural network

        Args:
            X(numpy.ndarray), shape (nx, m):
            contains the input data
            - nx is the number of input features to the neuron
            - m is the number of examples

            Y(numpy.ndarray), shape (1, m):
            contains the correct labels for the input data

            A1: the output of the hidden layer

            A2: the predicted output

            alpha: the learning rate

        * Updates the private attributes __W1, __b1, __W2, __b2 *
        """
        number_examples = X.shape[1]

        dZ2 = A2 - Y
        dW2 = np.matmul(dZ2, A1.T) / number_examples
        db2 = np.sum(dZ2, axis=1, keepdims=True) / number_examples
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2

        # derivative of a sigmoid activation function
        dA1 = A1 * (1 - A1)
        dZ1 = np.matmul(self.__W2.T, dZ2) * dA1
        dW1 = np.matmul(dZ1, X.T) / number_examples
        db1 = np.sum(dZ1, axis=1, keepdims=True) / number_examples
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Method:
            Trains the the neural network

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

        # for _ in range(iterations):
        #     A1, A2 = self.forward_prop(X)
        #     self.gradient_descent(X, Y, A1, A2, alpha=alpha)

        # prediction, cost = self.evaluate(X, Y)

        # return prediction, cost
        plot_cost = []
        for i in range(iterations + 1):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.A1, self.A2, alpha=alpha)

            if verbose == True or graph == True:
                if type(step) != int:
                    raise TypeError("step must be integer")
                if step < 0 or step > iterations:
                    raise ValueError("step must be positive and <= iterations")
                cost = self.cost(Y, self.A2)
                plot_cost.append(cost)
                if step:
                    if verbose and i % 100 == 0:
                        print("Cost after {} iterations: {}".format(i, cost))
        if graph:
            plt.plot(np.arange(0, iterations + 1), plot_cost, 'b-')
            plt.xlabel('iteration')
            plt.xticks(np.arange(0, iterations + 1, 500))
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        prediction, cost = self.evaluate(X, Y)

        return prediction, cost