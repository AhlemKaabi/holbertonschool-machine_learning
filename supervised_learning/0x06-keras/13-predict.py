#!/usr/bin/env python3
"""
    Keras - Predict
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Method:
        makes a prediction using a neural network:.

    Parameters:
        @network: the network model to test.
        @data: the input data to make the prediction with.
        @verbose: boolean that determines if output should
            be printed during the testing process.

    Returns:
        the prediction for the data
    """
    return network.predict(x=data, verbose=verbose)
