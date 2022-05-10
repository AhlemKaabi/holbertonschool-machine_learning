#!/usr/bin/env python3

from tensorflow.keras.layers import Embedding


# https://github.com/RaRe-Technologies/gensim/wiki/Using-Gensim-Embeddings-with-Keras-and-Tensorflow
def gensim_to_keras(model):
    """
    Method to convert a gensim word2vec model to a keras Embedding layer.

    Parameters
    ----------
    model : trained gensim word2vec models

    Returns
    -------
    trainable keras Embedding

    """
    keyed_vectors = model.wv
    # structure holding the result of training
    weights = keyed_vectors.vectors
    # vectors themselves, a 2D numpy array

    layer = Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=False,
    )
    return layer
