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

    # https://docs.w3cub.com/tensorflow~python/meta_graph
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + ".meta")
        saver.restore(sess, save_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection("accuracy")[0]
        prediction = sess.run(y_pred, feed_dict={x: X, y: Y})
        loss = sess.run(loss, feed_dict={x: X, y: Y})
        accuracy = sess.run(accuracy, feed_dict={x: X, y: Y})
    #     saver = tf.train.import_meta_graph(save_path + ".meta")
    #     saver.restore(session, save_path)
    #     # graph = tf.get_default_graph()
    #     # x = graph.get_operation_by_name('x').outputs[0]
    #     # y = graph.get_operation_by_name('y').outputs[0]

    #     x = tf.get_collection('x')[0]
    #     y = tf.get_collection('y')[0]

    #     y_pred = tf.get_collection('y_pred')[0]
    #     accuracy = tf.get_collection('accuracy')[0]
    #     loss = tf.get_collection('loss')[0]

    # #     y_pred = graph.get_operation_by_name('y_pred').outputs[0]
    # #     accuracy = graph.get_operation_by_name('accuracy').outputs[0]
    # #     loss = graph.get_operation_by_name('loss').outputs[0]
    # #     # restore() method runs the ops added by the constructor
    # #     # for restoring variables.
    # #     saver.restore(session, save_path)
    #     prediction = session.run(y_pred, feed_dict={x: X, y: Y})
    #     accuracy = session.run(accuracy, feed_dict={x: X, y: Y})
    #     loss = session.run(loss, feed_dict={x: X, y: Y})
    return prediction, accuracy, loss
