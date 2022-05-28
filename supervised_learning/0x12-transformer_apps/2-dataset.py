#!/usr/bin/env python3
""" Transformer Applications - TF Encode """
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

class Dataset():
    """ Dataset loads and preps a dataset for machine translation """

    def __init__(self):
        """ Class constructor """
        # https://youtu.be/YrMy-BAqk8k
        data_train, data_valid  = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split=["train", "validation"],
            as_supervised=True
        )
        # as_supervised=True: return tuple(img, lable), False return dict.
        # we can add with_info=True

        tokenizer_pt, tokenizer_en = self.tokenize_dataset(data_train)
        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en

        # update the data_train and data_validate attributes
        # by tokenizing the examples
        self.data_train = data_train.map(self.tf_encode)
        self.data_valid = data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        Method:
        -------
            Creates sub-word tokenizers for our dataset

        Parameters:
        -----------
            data (tuple): formatted as (pt, en)
                pt: the tf.Tensor containing the Portuguese sentence
                en: the tf.Tensor containing the corresponding English
                sentence

        Returns:
        --------
            tokenizer_pt: the Portuguese tokenizer
            tokenizer_en: the English tokenizer
        """
        text_Encoder = tfds.deprecated.text.subword_text_encoder.SubwordTextEncoder
        # then use build_from_corpus()
        # https://www.tensorflow.org/datasets/api_docs/python/tfds/deprecated/text/SubwordTextEncoder#build_from_corpus

        # generator yielding str, from which subwords will be constructed.
        # what is generator? ->
        # https://stackoverflow.com/questions/59576060/subwordtextencoder-build-from-corpus-splits-tokens-in-reserved-tokens
        # https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do

          # trying: corpus_generator_en = (en for _, en in data_train)

          # but: TypeError -> Expected binary or unicode string,
          # got <tf.Tensor: shape=(), dtype=string, numpy=b'and when you improve
        # searchability , you actually take away the one advantage of print ,
         # which is serendipity .'>

        # corpus_generator_en = (en.as_numpy() for _, en in data_train)
        # corpus_generator_pt = (pt.as_numpy_iterator() for pt,_ in data_train)
        # error:
        # 'tensorflow.python.framework.ops.EagerTensor' object
        # has no attribute 'as_numpy'
        # solution -> useful function, numpy!!!!! already we have a binary
        # string ;)
        # https://www.tensorflow.org/api_docs/python/tf/Tensor#expandable-1
        corpus_generator_en = (en.numpy() for _, en in data)
        corpus_generator_pt = (pt.numpy() for pt, _ in data)

        target_vocab_size = 2**15
        tokenizer_en = text_Encoder.build_from_corpus(
            corpus_generator_en,
            target_vocab_size
        )

        tokenizer_pt = text_Encoder.build_from_corpus(
            corpus_generator_pt,
            target_vocab_size
        )
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Method:
        -------
            Encodes a translation into tokens.(encode sub-word into tokens)

        Parameters:
        -----------
            pt (tf.Tensor):  the Portuguese sentence.
            en (tf.Tensor): the corresponding English sentence.

        Returns:
        --------
            tokenizer_pt (np.ndarray): the Portuguese tokenizer
            tokenizer_en (np.ndarray): the English tokenizer

        ** The tokenized sentences should include the start and **
                        end of sentence tokens

        ** The start token should be indexed as vocab_size **

        ** The end token should be indexed as vocab_size + 1 **
        """
        en = self.tokenizer_en.encode(en.numpy())
        vocab_size_en = self.tokenizer_en.vocab_size
        en.insert(0, vocab_size_en)
        en.append(vocab_size_en + 1)

        pt = self.tokenizer_pt.encode(pt.numpy())
        vocab_size_pt = self.tokenizer_pt.vocab_size
        pt.insert(0, vocab_size_pt)
        pt.append(vocab_size_pt + 1)

        return pt, en

    def tf_encode(self, pt, en):
        """
        Method:
        -------
            Acts as a tensorflow wrapper for the encode instance method.

        Parameters:
        -----------
            pt (tf.Tensor):  the Portuguese sentence
            en (tf.Tensor): the corresponding English sentence

        Returns:
        --------
            tensors

        ** Make sure to set the shape of the pt and en return tensors **
        """
        pt, en = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=tf.int64
        )
        return pt, en