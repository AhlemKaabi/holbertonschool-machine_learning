#!/usr/bin/env python3
"""
    Train
"""
import tensorflow.compat.v1 as tf
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Method:
         builds, trains, and saves a neural network classifier:

    Parameters:
        @X_train (numpy.ndarray): training input data.
        @Y_train (numpy.ndarray):training labels.
        @X_valid (numpy.ndarray): validation input data.
        @Y_valid (numpy.ndarray):  validation labels.
        @layer_sizes (array): list containing the number of nodes
            in each layer of the network.
        @activations (array): list containing the activation
            functions for each layer of the network.
        @alpha (float): learning rate
        @iterations (int): number of iterations to train over
        @save_path (string): where to save the model.


    Returns:
        the path where the model was saved.
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)
    tf.add_to_collection("y_pred", y_pred)
    tf.add_to_collection("loss", loss)
    tf.add_to_collection("accuracy", accuracy)
    tf.add_to_collection("train_op", train_op)
    init = tf.initializers.global_variables()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations + 1):
            t_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            t_acc = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            v_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            v_acc = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})
            if i == iterations or i % 100 == 0:
                print('After {} iterations:'.format(i))
                print('\tTraining Cost: {}'.format(t_cost))
                print('\tTraining Accuracy: {}'.format(t_acc))
                print('\tValidation Cost: {}'.format(v_cost))
                print('\tValidation Accuracy: {}'.format(v_acc))
        return saver.save(sess, save_path)