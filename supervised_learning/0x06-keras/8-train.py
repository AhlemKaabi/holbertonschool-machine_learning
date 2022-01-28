#!/usr/bin/env python3
"""
    Keras - Save the Best
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False):
    """
    Method:
        That trains a model using mini-batch gradient descent.
        Train the model with learning rate decay.
        save the best iteration of the model.

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

        @save_best: boolean indicating whether to save the model
        after each epoch if it is the best

        @filepath: file path where the model should be saved

    Returns:
         the one-hot matrix
    """
    def scheduler(epochs):
        """
        update the learning rate
        """
        # Time-based decay
        # https://neptune.ai/blog/how-to-choose-a-learning-rate-scheduler#implementation
        # decay = alpha / epochs ==> decay_rate
        return alpha / (1 + decay_rate * epochs)

    callbacks = []

    if validation_data:
        if early_stopping:
            callbacks.append(K.callbacks.EarlyStopping(patience=patience))

        if learning_rate_decay:
            # verbose: int. 0: quiet, 1: update messages.
            callbacks.append(K.callbacks.LearningRateScheduler(scheduler,
                                                               verbose=1))
    if save_best:
        # https://towardsdatascience.com/keras-callbacks-and-how-to-save-your-model-from-overtraining-244fc1de8608
        # a model is considered the best if its validation loss
        # is the lowest that the model has obtained
        # https://keras.io/api/callbacks/model_checkpoint/
        # default parameters
        checkpoint = K.callbacks.ModelCheckpoint(filepath=filepath,
                                                 monitor='val_loss',
                                                 verbose=0,
                                                 save_best_only=True,
                                                 mode='auto')
        callbacks.append(checkpoint)

    History = network.fit(x=data, y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=callbacks)

    return History
