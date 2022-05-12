#!/usr/bin/env python3
"""
    Natural Language Processing - Evaluation Metrics
    Unigram BLEU score
"""
import numpy as np


def uni_bleu(references, sentence):
    """
    Method:
    -------
        calculate the unigram BLEU score for a sentence.

    Parameters:
    -----------
        references: (list) of reference translations. each reference
          translation is a list of the words in the translation

          sentence: (list) containing the model proposed sentence.

    Returns:
    --------
        the unigram BLEU score.
    """
    #  modified unigram precision = m_max / w_t
    # m_max: maximum total count of unigrams in the candidate translation.
    # w_t: total number of unigrams in the candidate translation.

    word_count = {}
    for ref in references:
        for word in sentence:
            m = ref.count(word)
            if word in word_count:
                if word_count[word] < m:
                    word_count.update({word: m})
            else:
                word_count.update({word: m})
    w_t = len(sentence)
    precision = sum(word_count.values()) / w_t
    # get the length of the best matching referance sentence
    # which has the closest length to the length of the sentence
    length_diff = np.abs((np.array([len(r) for r in references])) - w_t)
    best_match_length = len(references[np.argmin(length_diff)])

    # brevity penalty factor

    if w_t > best_match_length:
        brevity_penality = 1
    else:
        brevity_penality = np.exp(1 - best_match_length / w_t)

    bleu_score = brevity_penality * precision

    return bleu_score
