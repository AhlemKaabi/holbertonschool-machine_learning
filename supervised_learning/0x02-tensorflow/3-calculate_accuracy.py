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
    # For each prediction, if the index with the largest value
    # matches the target value, then the prediction was correct.
    # https://jaredwinick.github.io/what_is_tf_keras/

    #  accuracy = correct_predictions / all_predictions ==> mean

# ---------
    # m = tf.keras.metrics.Accuracy()
    # m.update_state(y_true=y, y_pred=y_pred)
    # accuracy = m.result()
# ---------

    compare_data_bool = tf.math.equal(y, y_pred)

    # Casts compare_data to to be float.
    compare_data_float = tf.dtypes.cast(compare_data_bool, "float")
    # extract accuracy
    accuracy = tf.math.reduce_mean(compare_data_float)

    return accuracy
