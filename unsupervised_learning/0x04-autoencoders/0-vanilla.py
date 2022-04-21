#!/usr/bin/env python3
"""
    Autoencoders - "Vanilla" Autoencoder
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
    input_encoder = keras.Input(shape=(input_dims,))

    encode_hidden = keras.layers.Dense(hidden_layers[0],
                             activation='relu')(input_encoder)
    for i in hidden_layers[1::]:
        encode_hidden = keras.layers.Dense(i, activation='relu')(encode_hidden)

    latent_space = keras.layers.Dense(latent_dims,
                                      activation='relu')(encode_hidden)

    encoder = keras.Model(input_encoder, latent_space)

	# decoder
    input_decoder = keras.Input(shape=(latent_dims,))

    decode_hidden = keras.layers.Dense(hidden_layers[-1],
                             activation='relu')(input_decoder)

    for i in hidden_layers[-2::-1]:
        decode_hidden = keras.layers.Dense(i, activation='relu')(decode_hidden)

    decode_hidden = keras.layers.Dense(input_dims,
                                       activation='sigmoid')(decode_hidden)

    decoder = keras.Model(input_decoder, decode_hidden)

	# autoencoder
    autoencoder = keras.Model(input_encoder, decoder(encoder(input_encoder)))
    autoencoder.compile(loss='binary_crossentropy', optimizer='adam')
    return encoder, decoder, autoencoder
