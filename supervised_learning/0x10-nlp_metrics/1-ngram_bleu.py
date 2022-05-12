#!/usr/bin/env python3
"""
    Natural Language Processing - Evaluation Metrics
    N-gram BLEU score
"""
import numpy as np


def n_gram_generator(sentence, n=2):
    """N-gram generator"""
    n_grams = []
    for i in range(len(sentence) - n + 1):
        n_grams.append(' '.join(sentence[i:i + n]))
    return n_grams


def ngram_bleu(references, sentence, n):
    """
    Method:
    -------
        calculate the n-gram BLEU score for a sentence.

    Parameters:
    -----------
        references: (list) of reference translations. each reference
          translation is a list of the words in the translation

          sentence: (list) containing the model proposed sentence.

        n: size of the n-gram to use for evaluation.

    Returns:
    --------
        the n-gram BLEU score.
    """
    n_grams = n_gram_generator(sentence, n)
    n_gram_ref = list(n_gram_generator(sen, n) for sen in references)
    word_count = {}
    for ref in n_gram_ref:
        for word in n_grams:
            m = ref.count(word)
            if word in word_count:
                if word_count[word] < m:
                    word_count.update({word: m})
            else:
                word_count.update({word: m})
    w_t = len(sentence)
    precision = sum(word_count.values()) / len(n_grams)
    length_diff = np.abs((np.array([len(r) for r in references])) - w_t)
    best_match_length = len(references[np.argmin(length_diff)])

    # brevity penalty factor

    if w_t < best_match_length:
        brevity_penality = np.exp(1 - best_match_length / w_t)
    else:
        brevity_penality = 1

    bleu_score = brevity_penality * precision

    return bleu_score
