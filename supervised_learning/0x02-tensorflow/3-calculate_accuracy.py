#!/usr/bin/env python3
"""
    Accuracy
"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
    Method:
        calculates the accuracy of a prediction.

    Parameters:
        @y (float32): placeholder for the labels of the input data
        @y_pred (tensor): the network's predictions.

    Returns:
        a tensor containing the decimal accuracy of the prediction
    """
    #  accuracy = correct_predictions / all_predictions
    # correctly predected

    m = tf.keras.metrics.Accuracy()

    m.update_state(y_true=y, y_pred=y_pred)

    accuracy = m.result()

    return accuracy
