#!/usr/bin/env python3
"""
    Natural Language Processing - Word Embeddings
                    Train Word2Vec
"""
from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5,
                   negative=5, cbow=True, iterations=5, seed=0,
                   workers=1):
    """
    Method to create and train a gensim word2vec model.

    Parameters:

        sentences: (list) of sentences to be trained on

        size: dimensionality of the embedding layer

        min_count:  minimum number of occurrences of a word for use in training

        window:  maximum distance between the current and
        predicted word within a sentence.

        negative: (size) of negative sampling

        cbow: (boolean) to determine the training type; True is for CBOW;
        False is for Skip-gram

        iterations: number of iterations to train over

        seed: seed for the random number generator

        workers: number of worker threads to train the model

    Returns: the trained model
    """
    model = Word2Vec(sentences=sentences,
                     size=size,
                     window=window,
                     min_count=min_count,
                     negative=negative,
                     seed=seed,
                     workers=workers,
                     sg=cbow)

    model.train([["hello", "world"]], total_examples=1, epochs=1)

    return model
