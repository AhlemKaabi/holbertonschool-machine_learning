#!/usr/bin/env python3
"""
    CNN - LeNet-5 (Tensorflow 1)
"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    Method:
        builds a modified version of the LeNet-5
          architecture using tensorflow

    Parameters:
        @x(tf.placeholder), shape (m, 28, 28, 1):
         containing the input images for the network
           - m: the number of images

        @y(tf.placeholder),shape (m, 10):
        containing the one-hot labels for the network
    Returns:
        - a tensor for the softmax activated output
        - a training operation that utilizes Adam optimization
            (with default hyperparameters)
        - a tensor for the loss of the netowrk
        - a tensor for the accuracy of the network
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0)

    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    conv1 = tf.layers.Conv2D(6, 5, padding='same', activation='relu',
                             kernel_initializer=init)(x)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool2 = tf.layers.MaxPooling2D(2, 2)(conv1)

    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    conv3 = tf.layers.Conv2D(16, 5, padding='valid', activation='relu',
                             kernel_initializer=init)(pool2)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool4 = tf.layers.MaxPooling2D(2, 2)(conv3)

    # activations are flattened and fed into fully connected layer
    flat5 = tf.layers.Flatten()(pool4)

    # Fully connected layer with 120 nodes
    FC6 = tf.layers.Dense(120, activation='relu',
                          kernel_initializer=init)(flat5)

    # Fully connected layer with 84 nodes
    FC7 = tf.layers.Dense(84, activation='relu', kernel_initializer=init)(FC6)

    # Fully connected softmax output layer with 10 nodes
    FC8 = tf.layers.Dense(10, kernel_initializer=init)(FC7)

    output = tf.nn.softmax(FC8)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=output)

    train = tf.train.AdamOptimizer().minimize(loss)

    # accuracy
    max_correct_predictions_index = tf.math.argmax(y, axis=1)
    max_output_predictions_index = tf.math.argmax(output, axis=1)
    compare_data = tf.math.equal(max_output_predictions_index,
                                 max_correct_predictions_index)
    compare_data_float = tf.dtypes.cast(compare_data, "float")
    accuracy = tf.math.reduce_mean(compare_data_float)

    return output, train, loss, accuracy
