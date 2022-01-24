#!/usr/bin/env python3
"""
    keras - Sequential
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Method:
        that builds a neural network with the Keras library.

    Parameters:
        @nx: is the number of input features to the network

          @layers: is a list containing the number of nodes in
              each layer of the network

          @activations: is a list containing the activation
              functions used for each layer of the network

          @lambtha: is the L2 regularization parameter

          @keep_prob: is the probability that a node will be
              kept for dropout

    Returns:
         the keras model
    """

    model = K.Sequential()

    for i in range(0, len(layers)):
        model.add(K.layers.Dense(layers[i], input_shape=(nx,),
                               activation=activations[i],
                               activity_regularizer=K.regularizers.l2(lambtha)))
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(keep_prob))

    return model