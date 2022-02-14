#!/usr/bin/env python3
"""


"""
import tensorflow.compat.v1 as tf
import numpy as np

def create_placeholders(nx, classes):
    """
        Method: that return two placeholders, x and y, for the
        neural network
        @nx: the number of feature columns in our data
        @classes: the number of classes in our classifier
        Returns: placeholders named x and y, respectively
            - x is the placeholder for the input data
            to the neural network
            - y is the placeholder for the one-hot
            labels for the input data
    """
    x = tf.placeholder(tf.float32,  shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32,  shape=(None, classes), name='y')

    return x, y

def create_batch_norm_layer(prev, n, activation):
    """
    Method:
        that creates a batch normalization layer for a
         neural network in tensorflow:

    Args:
        @prev is the activated output of the previous layer
        @n is the number of nodes in the layer to be created
        @activation is the activation function that should be
        used on the output of the layer

    Returns:
        a tensor of the activated output for the layer
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    layer = tf.keras.layers.Dense(units=n, kernel_initializer=initializer)

    # Calculate the mean and variance of layer
    # for simple batch normalization pass axes=[0] (batch only).
    mean, variance = tf.nn.moments(layer(prev), axes=[0])

    # gamma and beta, initialized as vectors of 1 and 0 respectively
    gamma = tf.ones([n])
    beta = tf.zeros([n])

    epsilon = 1e-8
    batch_normalization_output = tf.nn.batch_normalization(layer(prev), mean,
                                                           variance, beta,
                                                           gamma, epsilon)
    return activation(batch_normalization_output)



def forward_prop(x, layer_sizes=[], activations=[]):
    #all layers get batch_normalization but the last one, that stays without any activation or normalization
    """
    Method:
        creates the forward propagation graph
        for the neural network.
    Parameters:
        @x (float32): the placeholder for the input data
        @layer_sizes (list): the number of nodes in each
            layer of the network
        @activations (list): the activation functions for each
            layer of the network
    Returns:
        the prediction of the network in tensor form
    """
    input_layer = x
    if len(layer_sizes) == len(activations):
        for i in range(len(layer_sizes) - 1):
            output_layer = create_batch_norm_layer(input_layer,
                                        layer_sizes[i],
                                        activations[i])
            input_layer = output_layer

        return output_layer
    return input_layer

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
    max_correct_predictions_index = tf.math.argmax(y, axis=1)
    max_output_predictions_index = tf.math.argmax(y_pred, axis=1)
    compare_data = tf.math.equal(max_output_predictions_index,
                                 max_correct_predictions_index)

    compare_data_float = tf.dtypes.cast(compare_data, "float")
    accuracy = tf.math.reduce_mean(compare_data_float)

    return accuracy

def calculate_loss(y, y_pred):
    """
    Method:
        calculates the softmax cross-entropy loss of a prediction.
    Parameters:
        @y (float32): placeholder for the labels of the input data
        @y_pred (tensor): the network's predictions.
    Returns:
        tensor containing the loss of the prediction.
    """
    return tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)

# training function allow for a smaller final batch (a.k.a. use the entire training set)
# def forward_prop(prev, layers, activations, epsilon):


def shuffle_data(X, Y):
    """
    shuffles the data points in two matrices the same way

    Parameters:
        X (numpy.ndarray): of shape(m, nx) to shuffle
            m: number of data points
            nx: number of features
        Y (numpy.ndarray): of shape(m, nx) to shuffle
            m: number of data points
            nx: number of features

    Returns:
         The shuffled X and Y matrices.
    """
    length = np.random.permutation(len(X))
    X_sh = X[length]
    Y_sh = Y[length]

    return X_sh, Y_sh

def create_train_op(loss, alpha, global_step):
    """
    Method:
        creates the training operation for the network.
    Parameters:
        @loss:  the loss of the network's prediction.
        @alpha: the learning rate.
    Returns:
        an operation that trains the network using
        gradient descent.
    """
    # hint: don't forget to add global_step parameter
    # in optimizer().minimize()
    opt = tf.train.GradientDescentOptimizer(learning_rate=alpha)

    return opt.minimize(loss)

def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
        beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
        save_path='/tmp/model.ckpt'):
    """
        Method:
            that builds, trains, and saves a neural network model in
            tensorflow using Adam optimization, mini-batch gradient descent,
              learning rate decay, and batch normalization:

        Args:
            @Data_train: a tuple containing the training inputs and training
               labels, respectively

            @Data_valid: a tuple containing the validation inputs and
              validation labels, respectively

            @layers: a list containing the number of nodes in each
              layer of the network

            @activation: a list containing the activation functions
               used for each layer of the network

            @alpha: the learning rate

            @beta1: the weight for the first moment of Adam Optimization

            @beta2: the weight for the second moment of Adam Optimization

            @epsilon: a small number used to avoid division by zero

            @decay_rate: the decay rate for inverse time decay of decay_stepsthe learning
               rate (the corresponding decay step should be 1)

            @batch_size: the number of data points that should be in a mini-batch

            @epochs: the number of times the training should pass through
               the whole dataset

            @save_path: the path where the model should be saved to

        Returns:
            The path where the model was saved.
    """
    # get X_train, Y_train, X_valid, and Y_valid from Data_train and Data_valid
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid
    # initialize x, y and add them to collection
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)
    # initialize y_pred and add it to collection
    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection("y_pred", y_pred)
    # intialize loss and add it to collection
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection("loss", loss)
    # intialize accuracy and add it to collection
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection("accuracy", accuracy)
    # intialize global_step variable
    global_step = decay_rate * epochs
    # hint: not trainable

    # compute decay_steps
    decay_steps = 0
    # create "alpha" the learning rate decay operation in tensorflow

    # initizalize train_op and add it to collection
    # hint: don't forget to add global_step parameter in optimizer().minimize()
    train_op = create_train_op(loss, alpha, global_step)
    tf.add_to_collection("train_op", train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
        train_accuracy = sess.run(accuracy, feed_dict={x: X_train,
                                                        y: Y_train})
        valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
        valid_accuracy = sess.run(accuracy, feed_dict={x: X_valid,
                                                           y: Y_valid})
        for i in range(epochs):
            # print training and validation cost and accuracy
            print('After {} iterations:'.format(i))
            print('\tTraining Cost: {}'.format(train_cost))
            print('\tTraining Accuracy: {}'.format(train_accuracy))
            print('\tValidation Cost: {}'.format(valid_cost))
            print('\tValidation Accuracy: {}'.format(valid_accuracy))
            # shuffle data
            X_shuffle, Y_shuffle = shuffle_data(X_train, Y_train)
            for j in range(0, X_train.shape[0], batch_size):
                # get X_batch and Y_batch from X_train shuffled and Y_train shuffled
                X_batch = X_shuffle[]
                Y_batch = Y_shuffle[]
                # run training operation
                sess.run(train_op, {x: X_batch, y: Y_batch})
                # print batch cost and accuracy
                loss_train = sess.run(loss, {x: X_batch, y: Y_batch})
                acc_train = sess.run(accuracy, {x: X_batch, y: Y_batch})
        # print training and validation cost and accuracy again
        print('After {} iterations:'.format(i))
        print('\tTraining Cost: {}'.format(train_cost))
        print('\tTraining Accuracy: {}'.format(train_accuracy))
        print('\tValidation Cost: {}'.format(valid_cost))
        print('\tValidation Accuracy: {}'.format(valid_accuracy))
        # save and return the path to where the model was saved
        saved_path = saver.save(sess, save_path)
        return saved_path
