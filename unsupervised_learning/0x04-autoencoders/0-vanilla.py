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
    input_en = keras.Input(shape=(input_dims,))
    enc = keras.layers.Dense(hidden_layers[0],
                             activation='relu')(input_en)
    for i in hidden_layers[1::]:
        enc = keras.layers.Dense(i, activation='relu')(enc)

    lt = keras.layers.Dense(latent_dims, activation='relu')(enc)

    encoder = keras.Model(input_en, lt)

    input_dec = keras.Input(shape=(latent_dims,))

    dec = keras.layers.Dense(hidden_layers[-1], activation='relu')(input_dec)

    for i in hidden_layers[-2::-1]:
        dec = keras.layers.Dense(i, activation='relu')(dec)
    dec = keras.layers.Dense(input_dims, activation='sigmoid')(dec)
    decoder = keras.Model(input_dec, dec)
    autoen = keras.Model(input_en, decoder(encoder(input_en)))
    autoen.compile(loss='binary_crossentropy', optimizer='adam')
    return encoder, decoder, autoen
