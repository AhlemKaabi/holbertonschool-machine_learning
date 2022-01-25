#!/usr/bin/env python3
"""
    Keras - Save and Load Weights
"""


def save_weights(network, filename, save_format='h5'):
    """
    Method:
        saves a model's weights.

    Parameters:
        @network: the model whose weights should be saved
        @filename: the path of the file that the weights
         should be saved to
        @save_format: the format in which the weights
         should be saved

    Returns: None
    """
    network.save_weights(filename)
    return None


def load_weights(network, filename):
    """
    Method:
        loads a model's weights.

    Parameters:
        @network: the model to which the weights should be loaded
        @filename is the path of the file that the weights
        should be loaded from

    Returns:
        None
    """
    network.load_weights(filename)
