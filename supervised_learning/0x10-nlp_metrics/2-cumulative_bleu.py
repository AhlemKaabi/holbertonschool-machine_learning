#!/usr/bin/env python3
"""
    Natural Language Processing - Evaluation Metrics
    Cumulative N-gram BLEU score
"""
import numpy as np


def n_gram_generator(sentence, n=2):
    """N-gram generator"""
    n_grams = []
    for i in range(len(sentence) - n + 1):
        n_grams.append(' '.join(sentence[i:i + n]))
    return n_grams


def cumulative_bleu(references, sentence, n):
    """
    Method:
    -------
        calculate the cumulative n-gram BLEU score for a sentence.

    Parameters:
    -----------
        references: (list) of reference translations. each reference
          translation is a list of the words in the translation

          sentence: (list) containing the model proposed sentence.

          n: size of the largest n-gram to use for evaluation.

    ** All n-gram scores should be weighted evenly **

    Returns:
    --------
        the cumulative n-gram BLEU score.
    """
    cumul_BLEU = []
    for i in range(1, n+1):
        n_grams = n_gram_generator(sentence, i)
        n_gram_ref = list(n_gram_generator(sen, i) for sen in references)
        word_count = {}
        for ref in n_gram_ref:
            for word in n_grams:
                m = ref.count(word)
                if word in word_count:
                    if word_count[word] < m:
                        word_count.update({word: m})
                else:
                    word_count.update({word: m})
        precision = sum(word_count.values()) / len(n_grams)
        cumul_BLEU.append(precision)

    w_t = len(sentence)
    length_diff = np.abs((np.array([len(r) for r in references])) - w_t)
    best_match_length = len(references[np.argmin(length_diff)])

    # brevity penalty factor

    if w_t < best_match_length:
        brevity_penality = np.exp(1 - best_match_length / w_t)
    else:
        brevity_penality = 1

    cumul_precision = np.exp(np.sum((1/n) * np.log(np.array(cumul_BLEU))))

    bleu_score = brevity_penality * cumul_precision

    return bleu_score
