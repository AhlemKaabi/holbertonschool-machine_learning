#!/usr/bin/env python3
"""
    Momentum
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Method:
        updates a variable using the gradient descent
        with momentum optimization algorithm:

    Args:
        @alpha: the learning rate
        @beta1: the momentum weight
        @var(numpy.ndarray) containing the variable to be updated
        @grad(numpy.ndarray): containing the gradient of var
        @v: the previous first moment of var

    Returns:
        the updated variable and the new moment, respectively.
    """
    W = var
    dW = grad
    VdW = v
    VdW = beta1 * VdW + (1 - beta1) * dW
    W = W - alpha * VdW
    return W, VdW
