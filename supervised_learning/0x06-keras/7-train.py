#!/usr/bin/env python3
"""
    Keras - Learning Rate Decay
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
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

        @validation_data: the data to validate the model with,
            if not None.

        @early_stopping: boolean that indicates whether early
              stopping should be used.

        @patience: the patience used for early stopping.

        @learning_rate_decay: boolean that indicates whether
          learning rate decay should be use.

          @alpha: the initial learning rate.

        @decay_rate: the decay rate.


    Returns:
         the one-hot matrix
    """
    def scheduler(epoch):
        """
        update the learning rate
        """
        # Time-based decay
        # https://neptune.ai/blog/how-to-choose-a-learning-rate-scheduler#implementation
        # decay = alpha / epochs ==> decay_rate
        return alpha / (1 + decay_rate * epoch)

    callbacks = []

    if validation_data:
        if early_stopping:
            callbacks.append(K.callbacks.EarlyStopping(patience=patience))

        if learning_rate_decay:
            # verbose: int. 0: quiet, 1: update messages.
            callbacks.append(K.callbacks.LearningRateScheduler(scheduler,
                                                               verbose=1))

    History = network.fit(x=data, y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=callbacks)
    return History
