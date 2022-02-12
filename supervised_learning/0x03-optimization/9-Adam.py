#!/usr/bin/env python3
"""
    Adam
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2,
                          epsilon, var, grad, v, s, t):
    """
    Method:
        That updates a variable in place using the Adam
         optimization algorithm.

    Args:
        @alpha: the learning rate
        @beta1: the weight used for the first moment
        @beta2: the weight used for the second moment
        @epsilon: a small number to avoid division by zero
        @var: a numpy.ndarray containing the variable to be updated
        @grad: a numpy.ndarray containing the gradient of var
        @v: the previous first moment of var
        @s: the previous second moment of var
        @t: the time step used for bias correction

    Returns:
        The updated variable, the new first moment, and
         the new second moment, respectively
    """
    W = var
    dW = grad

    VdW = v
    VdW = beta1 * VdW + (1 - beta1) * dW
    c_VdW = VdW / (1 - beta1 ** t)

    SdW = s
    SdW = beta2 * SdW + (1 - beta2) * (dW ** 2)
    c_SdW = SdW / (1 - beta2 ** t)

    W = W - alpha * c_VdW / (np.sqrt(c_SdW) + epsilon)

    return W, c_VdW, c_SdW
