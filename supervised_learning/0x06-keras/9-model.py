#!/usr/bin/env python3
"""
    Keras - Save and Load Model
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Method:
        saves an entire model.

    Parameters:
        @network: the model to save
        @filename: the path of the file that the model
        should be saved to

    Returns: None
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    Method:
        loads an entire model.

    Parametres:
        @filename: the path of the file that
        the model should be loaded from

    Returns: the loaded model
    """
    return K.models.load_model(filename)
