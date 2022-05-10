#!/usr/bin/env python3
"""
    Natural Language Processing - Word Embeddings
                        TF-IDF
    TF (Term Frequency) - IDF (Inverse Documet Frequency)
"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Method to create  a TF-IDF embedding matrix.

    Parameters:
        sentences: (list) sentences to analyze.
        vocab: (list) the vocabulary words to use for the analysis.
            - If None, all words within sentences should be used

    Returns: embeddings, features
        embeddings (numpy.ndarray of shape (s, f)):
            containing the embeddings
            - s: the number of sentences in `sentences`.
            - f: the number of features analyzed.
        features: (list) the features used for embeddings.
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    features = vectorizer.get_feature_names()
    embeddings = X.toarray()
    return embeddings, features
