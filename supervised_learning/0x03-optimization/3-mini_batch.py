#!/usr/bin/env python3
"""
    Mini-Batch
"""
import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train,
                     X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Method:
        Trains a loaded neural network model using
        mini-batch gradient descent:

    Args:
        @X_train(numpy.ndarray),shape (m, 784)
          containing the training data
        - m: the number of data points
        - 784: the number of input features

        @Y_train: one-hot (numpy.ndarray),shape (m, 10)
           containing the training labels
        - 10: the number of classes the model should classify

        @X_valid: (numpy.ndarray), shape (m, 784)
            containing the validation data

        @Y_valid: one-hot (numpy.ndarray),hape (m, 10)
         containing the validation labels

        @batch_size: the number of data points in a batch

        @epochs: number of times the training should pass
            through the whole dataset

        @load_path: the path from which to load the model

        @save_path: the path to where the model should be
           saved after training

    Returns:
        the path where the model was saved
    """
    m = X_train.shape[0]
    with tf.Session() as sess:
        # import meta graph and restore session
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)
        # Get the following tensors and ops from the collection restored
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        train_op = tf.get_collection('train_op')[0]
        # loop over epochs - passes throught the training set
        for i in range(epochs):
            # shuffle data
            X_shuffle, Y_shuffle = shuffle_data(X_train, Y_train)
            # Before the first epoch and after every subsequent epoch
            # should be printed
            train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            train_accuracy = sess.run(accuracy, feed_dict={x: X_train,
                                                           y: Y_train})
            valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            valid_accuracy = sess.run(accuracy, feed_dict={x: X_valid,
                                                           y: Y_valid})
            # print
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))
            if i != epochs:
                # get the number of iterations
                iterations = m // batch_size
                start_batch = 0
                end_batch = batch_size
                for j in range(iterations):
                    X_batch = X_shuffle[start_batch:end_batch]
                    Y_batch = Y_shuffle[start_batch:end_batch]
                    sess.run(train_op, {x: X_batch, y: Y_batch})
                    loss_train = sess.run(loss, {x: X_batch, y: Y_batch})
                    acc_train = sess.run(accuracy, {x: X_batch, y: Y_batch})
                    start_batch += batch_size
                    end_batch += batch_size
                    if (j) % 100 == 0 and j != 0:
                        print('\tStep {}:'.format(j))
                        print('\t\tCost: {}'.format(loss_train))
                        print('\t\tAccuracy: {}'.format(acc_train))
        # Save session
        save = saver.save(sess, save_path)
        return save
