#!/usr/bin/env python3
"""
    keras - Input
    Not allowed to use the Sequential class!
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

    reg = K.regularizers.l2(lambtha)

    inputs = K.Input(shape=(nx,))
    hidden_layer = inputs
    for i in range(0, len(layers)):
        if i == len(layers) - 1:
            outputs = K.layers.Dense(layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=reg)(hidden_layer)

        hidden_layer = K.layers.Dense(layers[i],
                                      activation=activations[i],
                                      kernel_regularizer=reg)(hidden_layer)
        hidden_layer = K.layers.Dropout(1 - keep_prob)(hidden_layer)

    model = K.Model(inputs=inputs, outputs=outputs)
    return model
