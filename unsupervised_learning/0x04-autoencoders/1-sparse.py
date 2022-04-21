#!/usr/bin/env python3
"""
    Autoencoders - Sparse Autoencoder
    -> A Regularized Autoencoder.
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Method to create a sparse autoencoder.

    Parameters:
        input_dims (integer):
         containing the dimensions of the model input

        hidden_layers (list):
         containing the number of nodes for each hidden layer in the encoder,
         respectively!

        latent_dims (integer):
         containing the dimensions of the latent space representation.

        lambtha: the regularization parameter used for L1 regularization on
           the encoded output

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
    Input = keras.layers.Input(shape=(input_dims,))

    # encoder Model
    # input layer (input_dims) => hidden layers => output layer (latent_dims)
    encode_hidden = Input
    for e in hidden_layers:
        encode_hidden = keras.layers.Dense(e, activation='relu')(encode_hidden)

    regularization = keras.regularizers.l1(lambtha)
    encode_output_hidden = keras.layers.Dense(latent_dims,
                                              activation='relu',
                                              kernel_regularizer=regularization
                                              )(encode_hidden)

    encoder = keras.Model(Input, encode_output_hidden)

    # decoder Model
    # input layer (latent_dims) =>
    #                     hidden layers (inversed) =>
    #                                     output layer (input_dims)
    reversed_list = hidden_layers[::-1]
    print(reversed_list)

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
