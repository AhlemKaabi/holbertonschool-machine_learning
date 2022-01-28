#!/usr/bin/env python3
"""
    Keras - Save and Load Configuration

    Quick and simple ways to save and load deep learning
    models using JSON and YAML files

    Saving the architecture

    The model's configuration (or architecture)
    specifies what layers the model contains,
    and how these layers are connected.

    this only applies to models defined using the
    functional or Sequential apis not subclassed models.
    https://www.tensorflow.org/guide/keras/save_and_serialize#saving_the_architecture

"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Method:
        saves a model's configuration in JSON format.

    Parameters:
        @network: the model whose weights should be saved
        @filename: the path of the file that the weights
        should be saved to

    Returns: None
    """
    model_json = network.to_json()
    json_file = open(filename, 'w')
    json_file.write(model_json)


def load_config(filename):
    """
    Method:
        loads a model with a specific configuration.

    Parameters:
        @filename is the path of the file that the weights
        should be loaded from

    Returns:
         the loaded model
    """

    json_file = open(filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = K.models.model_from_json(loaded_model_json)
    return loaded_model
