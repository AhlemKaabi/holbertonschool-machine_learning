#!/usr/bin/env python3
"""
    Keras - Optimize
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Method:
        That sets up Adam optimization for a keras model
        with categorical crossentropy loss and accuracy metrics.

    Parameters:
        @network: is the model to optimize
        @alpha: is the learning rate
        @beta1: is the first Adam optimization parameter
        @beta2: is the second Adam optimization parameter

    Returns:
        None
    """
    # https://www.tensorflow.org/api_docs/python/tf/keras/Model
    # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
    opt = K.optimizers.Adam(learning_rate=alpha,
                            beta_1=beta1,
                            beta_2=beta2)
    network.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])
    return None
