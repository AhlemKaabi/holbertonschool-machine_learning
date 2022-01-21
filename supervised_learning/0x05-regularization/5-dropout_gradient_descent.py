#!/usr/bin/env python3
"""
	Gradient Descent with Dropout
"""


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
	Method:
		updates the weights of a neural network with Dropout
  		regularization using gradient descent:

    Parameters:
		@Y:  one-hot numpy.ndarray  contains the correct
  			labels for the data.
		@weights: a dictionary of the weights and biases
  			of the neural network.
		@cache: a dictionary of the outputs and dropout masks of
  			each layer of the neural network.
  		@alpha: the learning rate
		@keep_prob: the probability that a node will be kept.
		@L: the number of layers of the network.

    """