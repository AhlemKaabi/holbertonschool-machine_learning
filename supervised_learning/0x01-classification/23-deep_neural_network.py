#!/usr/bin/env python3
"""
    Class DeepNeuralNetwork that defines a deep neural network
     performing binary classification.
"""
import numpy as np
import matplotlib.pyplot as plt


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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
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

    @property
    def L(self):
        """ get the L """
        return self.__L

    @property
    def cache(self):
        """ get the cache """
        return self.__cache

    @property
    def weights(self):
        """
        get the wights
        """
        return self.__weights

    def forward_prop(self, X):
        """
        Method:
            Calculates the forward propagation of the neural network
            Updates the private attribute __cache
            The neurons should use a sigmoid activation function.
            The activated outputs of each layer should be saved
            in the __cache dictionary
        Args:
            X (numpy.ndarray): shape (nx, m)
                nx : he number of input features to the neuron
                m : the number of examples
        Returns:
            the output of the neural network and the cache
        """
        self.cache['A0'] = X
        for l_ in range(self.L):
            W_l = self.weights['W' + str(l_ + 1)]
            b_l = self.weights['b' + str(l_)]
            A_prev = self.cache['A' + str(l_)]
            Z1 = np.matmul(W_l, A_prev) + b_l
            A_l = 1 / (1 + np.exp(-Z1))
            self.cache['A' + str(l_ + 1)] = A_l
        A = self.cache['A' + str(self.L)]
        return A, self.cache

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
        output, _ = self.forward_prop(X)
        # Where True, yield x, otherwise yield y.
        prediction = np.where(output >= 0.5, 1, 0)
        cost = self.cost(Y, output)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Method:
            Calculates *one pass* of gradient descent on
            the neural network

        Args:
            Y(numpy.ndarray), shape (1, m):
            contains the correct labels for the input data

            cache(dictionary): containing all the intermediary
            values of the network

            alpha: the learning rate

        * Updates the private attribute __weights *
        """
        m = Y.shape[1]
        # Initialization for backpropagation algorithm
        dZ = cache['A' + str(self.L)] - Y

        for i in range(self.__L, 0, -1):
            A = cache['A' + str(i - 1)]

            dW = (1 / m) * np.matmul(A, dZ.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            # derivative of a sigmoid activation function
            dA1 = A * (1 - A)
            W = self.__weights['W' + str(i)]
            dZ = np.matmul(W.T, dZ) * dA1

            self.__weights['W' + str(i)] -= (alpha * dW).T
            self.__weights['b' + str(i - 1)] -= alpha * db

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

        plot_cost = []
        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

            if verbose == True or graph == True:
                if type(step) != int:
                    raise TypeError("step must be integer")
                if step < 0 or step > iterations:
                    raise ValueError("step must be positive and <= iterations")
                cost = self.cost(Y, A)
                plot_cost.append(cost)
                if step:
                    if verbose and i % 100 == 0:
                        print("Cost after {} iterations: {}".format(i, cost))
        if graph:
            plt.plot(np.arange(0, iterations + 1), plot_cost, 'b-')
            plt.xlabel('iteration')
            plt.xticks(np.arange(0, iterations + 1, 1000))
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        prediction, cost = self.evaluate(X, Y)

        return prediction, cost
