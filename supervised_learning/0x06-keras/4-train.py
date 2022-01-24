#!/usr/bin/env python3
"""
    Keras - Train
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """
    Method:
        That trains a model using mini-batch gradient descent.

    Parameters:
        @network: is the model to train

          @data: (numpy.ndarray) of shape (m, nx) containing
            the input data
        @labels: (one-hot numpy.ndarray) of shape (m, classes)
              containing the labels of data

          @batch_size: the size of the batch used
            for mini-batch gradient descent

        @epochs: the number of passes through data for
              mini-batch gradient descent.

        @verbose: boolean that determines if output
              should be printed during training

        @shuffle: boolean that determines whether to shuffle
              the batches every epoch.

    Returns:
         the one-hot matrix
    """

    History = network.fit(x=data, y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle)
    return History
