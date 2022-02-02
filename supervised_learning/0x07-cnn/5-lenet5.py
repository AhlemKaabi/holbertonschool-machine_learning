#!/usr/bin/env python3
"""
    CNN - LeNet-5 (Keras)
"""
import tensorflow.keras as K


def lenet5(X):
    """
    Method :
        builds a modified version of the LeNet-5
          architecture using keras

    Parameters:
        @X(K.Input), shape(m, 28, 28, 1)
         containing the input images for the network
        - m: the number of images

     Returns:
        K.Model compiled to use Adam optimization
          (with default hyperparameters) and accuracy metrics
    """
    init = K.initializers.HeNormal()

    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    conv1 = K.layers.Conv2D(6, 5, padding='same', activation='relu',
                            kernel_initializer=init)(X)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool2 = K.layers.MaxPooling2D(2, 2)(conv1)

    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    conv3 = K.layers.Conv2D(16, 5, padding='valid', activation='relu',
                            kernel_initializer=init)(pool2)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool4 = K.layers.MaxPooling2D(2, 2)(conv3)

    # activations are flattened and fed into fully connected layer
    flat5 = K.layers.Flatten()(pool4)

    # Fully connected layer with 120 nodes
    FC6 = K.layers.Dense(120, activation='relu',
                         kernel_initializer=init)(flat5)

    # Fully connected layer with 84 nodes
    FC7 = K.layers.Dense(84, activation='relu', kernel_initializer=init)(FC6)

    # Fully connected softmax output layer with 10 nodes
    FC_output = K.layers.Dense(10, kernel_initializer=init,
                               activation='softmax')(FC7)

    model = K.models.Model(X, FC_output)

    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
