#!/usr/bin/env python3
"""
    Evaluate
"""
import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """
    Method:
        Evaluates the output of a neural network.

    Parameters:
        @X (numpy.ndarray) containing the input data to evaluate
        @Y (numpy.ndarray) containing the one-hot labels for X
        @save_path is the location to load the model from

        Returns:
            the network's prediction, accuracy, and loss,
            respectively
    """
    # tf.reset_default_graph()

    # saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    # https://docs.w3cub.com/tensorflow~python/meta_graph
    with tf.Session() as session:
        session.run(init)
        saver = tf.train.import_meta_graph(save_path + ".meta")

        saver.restore(session, save_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]

        prediction = session.run(y_pred, feed_dict={x: X, y: Y})
        acc_eval = session.run(accuracy, feed_dict={x: X, y: Y})
        loss_eval = session.run(loss, feed_dict={x: X, y: Y})
        return prediction, acc_eval, loss_eval
