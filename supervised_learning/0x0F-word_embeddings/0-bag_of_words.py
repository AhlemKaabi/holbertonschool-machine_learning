#!/usr/bin/env python3
"""
    Natural Language Processing - Word Embeddings
                    Bag Of Words
"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    Method to create a bag of words embedding matrix.

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
    # https://www.analyticsvidhya.com/blog/2021/08/a-friendly-guide-to-nlp-bag-of-words-with-python-example/
    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    # https://stackoverflow.com/questions/70640923/countvectorizer-object-has-no-attribute-get-feature-names-out
    features = vectorizer.get_feature_names()
    embeddings = X.toarray()
    return embeddings, features
