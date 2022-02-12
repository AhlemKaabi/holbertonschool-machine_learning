"""
    RMSProp - Root Mean Squared Propagation
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Method:
        That updates a variable using
         the RMSProp optimization algorithm.

    Args:

        @alpha: the learning rate
        @beta2 the RMSProp weight
        @epsilon: a small number to avoid division by zero
        @var(numpy.ndarray) containing the variable to be updated
        @grad(numpy.ndarray) containing the gradient of var
        @s: the previous second moment of var

    Returns:
        The updated variable and the new moment, respectively
    """
    W = var
    dW = grad
    SdW = 0
    SdW = beta2 * SdW + (1 - beta2) * (dW ** 2)
    W = W - alpha * (dW / (epsilon + np.sqrt(SdW)))
    return W, SdW
