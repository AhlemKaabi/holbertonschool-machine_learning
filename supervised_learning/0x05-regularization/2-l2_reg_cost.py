#!/usr/bin/env python3
"""
	L2 Regularization Cost / tensorFlow
"""
import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """
	Method:
		calculates the cost of a neural network with
  		L2 regularization.

    Parameters:
		@cost: a tensor containing the cost of the network
  			without L2 regularization

	Returns:
		a tensor containing the cost of the network
  		accounting for L2 regularization
    """
    # tf.losses.get_regularization_losses
    # Returns: A list of regularization losses as Tensors.
    return cost + tf.losses.get_regularization_losses()