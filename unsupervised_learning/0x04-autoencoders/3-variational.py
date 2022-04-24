#!/usr/bin/env python3
"""
    Autoencoders - Variational Autoencoder(VAE)
    -> generative model.
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

    **
    All layers should use a relu activation
    except for the mean and log variance layers in the encoder,
    which should use None, and the last layer in the decoder,
    which should use sigmoid
    **
    """
    # https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf
    # https://blog.keras.io/building-autoencoders-in-keras.html
    # autoencoder(784, [512], 2)
    inputs = keras.layers.Input(shape=(input_dims,))

    encode_hidden = inputs
    for h in hidden_layers:
        encode_hidden = keras.layers.Dense(h, activation='relu')(encode_hidden)

    z_mean = keras.layers.Dense(latent_dims)(encode_hidden)

    z_log_sigma = keras.layers.Dense(latent_dims)(encode_hidden)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = keras.backend.random_normal(
            shape=(keras.backend.shape(z_mean)[0],
                   latent_dims),
            mean=0,
            stddev=1)
        return z_mean + keras.backend.exp(z_log_sigma / 2) * epsilon

    Z_layer = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])

    # Create encoder
    encoder = keras.Model(inputs,
                          [z_mean, z_log_sigma, Z_layer],
                          name='encoder')
    # Create decoder
    latent_inputs = keras.Input(shape=(latent_dims,), name='z_sampling')
    x = latent_inputs
    for h in hidden_layers[::-1]:
        x = keras.layers.Dense(h, activation='relu')(x)

    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(latent_inputs, decoder_output, name='decoder')
    # instantiate VAE model
    outputs = decoder(encoder(inputs)[-1])
    vae = keras.Model(inputs, outputs, name='vae_mlp')

    def loss_function(inputs, outputs, input_dims, z_mean, z_log_sigma):
        """
        loss function
        """
        def loss(inputs, outputs):
            """
            loss
            """
            reconstruction_loss = keras.losses.binary_crossentropy(inputs,
                                                                   outputs)
            reconstruction_loss *= input_dims

            sq_mean = keras.backend.square(z_mean)
            ex_sig = keras.backend.exp(z_log_sigma)
            kl_loss = 1 + z_log_sigma - sq_mean - ex_sig
            kl_loss = keras.backend.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
            return(vae_loss)
        return loss

    vae.compile(optimizer='adam', loss=loss_function(inputs, outputs,
                                                     input_dims, z_mean,
                                                     z_log_sigma))
    return encoder, decoder, vae
