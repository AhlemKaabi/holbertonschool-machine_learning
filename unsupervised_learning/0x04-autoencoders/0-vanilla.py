#!/usr/bin/env python3
"""
    Autoencoders - Vanilla Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Method to create an autoencoder.

    Parameters:
        input_dims (integer):
         containing the dimensions of the model input

        hidden_layers (list):
         containing the number of nodes for each hidden layer in the encoder,
         respectively!

        latent_dims (integer):
         containing the dimensions of the latent space representation.

    Returns: encoder, decoder, auto

        encoder: the encoder model.
        decoder: the decoder model.
        auto: the full autoencoder model.

    ** The hidden layers should be reversed for the decoder **

    ** The autoencoder model should be compiled using adam optimization and
    binary cross-entropy loss **

    ** All layers should use a relu activation except for the last layer in the
    decoder, which should use sigmoid **

    """
    # encoder
    Input = keras.layers.Input(shape=(input_dims,))

    encode_hidden = Input
    for e in hidden_layers:
        encode_hidden = keras.layers.Dense(e, activation='relu')(encode_hidden)

    encode_output_hidden = keras.layers.Dense(latent_dims,
                                              activation='relu')(encode_hidden)

    encoder = keras.Model(Input, encode_output_hidden)

    # decoder
    reversed_list = hidden_layers[::-1]

    decode_hidden_input = keras.layers.Input(shape=(latent_dims,))
    decode_hidden = decode_hidden_input

    for d in reversed_list:
        decode_hidden = keras.layers.Dense(d, activation='relu')(decode_hidden)

    decode_output = keras.layers.Dense(input_dims,
                                       activation='sigmoid')(decode_hidden)

    decoder = keras.Model(decode_hidden_input, decode_output)

    autoencoder = keras.Model(Input, decoder(encoder(Input)))
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
