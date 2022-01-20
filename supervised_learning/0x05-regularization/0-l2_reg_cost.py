#!/usr/bin/env python3
"""
    L2 Regularization Cost / numpy
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Method:
        calculates the cost of a neural network with
          L2 regularization.

    Parameters:
        @cost: cost of the network without L2 regularization.
        @lambtha: the regularization parameter.
        @weights: dictionary of the weights and biases.
        @L: the number of layers in the neural network.
        @m: the number of data points used.

    Returns:
        cost of the network accounting for L2 regularization
    """

    # cost = loss + L2 Regularization parameter

    Forbenius_norm = 0
    for weight in weights.values():
        Forbenius_norm += np.linalg.norm(weight) ** 2
    L2_reg_param = (lambtha / (2 * m) )* Forbenius_norm

    L2_reg_cost = cost + L2_reg_param
    return L2_reg_cost

