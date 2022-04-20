#!/usr/bin/env python3
"""
    Autoencoders - Convolutional Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Method to create a convolutional autoencoder.

    Parameters:
        input_dims (tuple of integers ):
         containing the dimensions of the model input.

        filters (list):
          list containing the number of filters for each convolutional layer
        in the encoder, respectively

        latent_dims (tuple of integers):
          containing the dimensions of the latent space representation.

    Returns: encoder, decoder, auto

        encoder: the encoder model.
        decoder: the decoder model.
        auto: the full autoencoder model.

    ** The filters should be reversed for the decoder **

    **
    Each convolution in the encoder should use a kernel size of (3, 3) with
    same padding and relu activation, followed by max pooling of size (2, 2)
    **

    **
    Each convolution in the decoder, except for the last two, should use
    a filter size of (3, 3) with same padding and relu activation,
    followed by upsampling of size (2, 2)

        - The second to last convolution should instead use valid padding.
        - The last convolution should have the same number of filters as the
        number of channels in input_dims with sigmoid activation and
        no upsampling.
    **

    ** The autoencoder model should be compiled using adam optimization and
    binary cross-entropy loss **
    """
    # https://www.youtube.com/watch?v=P2lYhhCZ0Vg
    Input = keras.layers.Input(shape=input_dims)

    # encoder
    encode_hidden = Input
    for k in filters:
        encode_hidden = keras.layers.Conv2D(k, 3,
                                            activation='relu',
                                            padding='same')(encode_hidden)
        encode_hidden = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                                  padding='same'
                                                  )(encode_hidden)

    encode_output_hidden = encode_hidden

    encoder = keras.Model(Input, encode_output_hidden)

    decode_hidden_input = keras.layers.Input(shape=latent_dims)
    decode_hidden = decode_hidden_input
    for k in filters[::-1][:-1]:
        decode_hidden = keras.layers.Conv2D(k, 3,
                                            activation='relu',
                                            padding='same')(decode_hidden)
        decode_hidden = keras.layers.UpSampling2D(size=(2, 2))(decode_hidden)
    # The second to last convolution
    decode_hidden = keras.layers.Conv2D(filters[0], 3,
                                        activation='relu',
                                        padding='valid')(decode_hidden)

    decode_hidden = keras.layers.UpSampling2D(size=(2, 2))(decode_hidden)

    # number of channels in the input_dims => reconstruct the image!
    Last_k = input_dims[2]
    decode_output_hidden = keras.layers.Conv2D(Last_k, 3,
                                               activation='sigmoid',
                                               padding='same')(decode_hidden)

    decoder = keras.Model(decode_hidden_input, decode_output_hidden)

    autoencoder = keras.Model(Input, decoder(encoder(Input)))
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
