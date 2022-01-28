#!/usr/bin/env python3
"""
    Keras - Test
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Method:
        that tests a neural network.

    Parameters:
        @network: the network model to test.
           @data: the input data to test the model with.
        @labels: he correct one-hot labels of data.
        @verbose: boolean that determines if output should
             be printed during the testing process.

    Returns:
         the loss and accuracy of the model with the
           testing data, respectively
    """
    # evaluate() :Returns the loss value & metrics values
    # for the model in test mode.
    return network.evaluate(x=data, y=labels, verbose=1)