# **Attention in Neural Networks**

sequace to secence model (RNN)

## **Learning Objectives**

* What is the attention mechanism?
	* Attention is proposed as a method to both `align` and `translate`.
	* [The Attention Mechanism from Scratch](https://machinelearningmastery.com/the-attention-mechanism-from-scratch/)
	* Attention mechanism was developed to improve the performance of the Encoder-Decoder RNN on machine translation. - [Link](https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/)
	* Instead of encoding the input sequence into a single fixed context vector, the attention model develops a context vector that is filtered specifically for each output time step.
* How to apply attention to RNNs
	* Encoding
	* Alignment
	* Weighting
	* Context Vector
	* Decode
* What is a transformer?
* How to create an encoder-decoder transformer model
* What is GPT?
* What is BERT?
* What is self-supervised learning?
* How to use BERT for specific NLP tasks
* What is SQuAD? GLUE?

## **Introduction**
* What is the Attention mechanism?
	* A method for determining which terms are most important in a sequence
* A Transformer:
	* Is a novel neural network
	* Utilizes the Attention mechanism
	* Utilizes Fully Connected Networks
	* Utilizes dropout
	* Utilizes layer normalization
* BERT was novel because:
	* It introduced self-supervised learning techniques
	* It can be fine tuned for various NLP tasks
* The database to use for Question-Answering is:
	* SQuAD
* Layer Normalization is different from Batch Normalization because:
	* It normalizes the layer output for each example instead of across the batch

## **Encoder-Decoder**

* Encoder: The encoder is responsible for stepping through the input time steps and encoding the entire sequence into a fixed length vector called a `context vector`.
* Decoder: The decoder is responsible for stepping through the output time steps while reading from the `context vector`.

> We propose a novel neural network architecture that learns to encode a variable-length sequence into a fixed-length vector representation and to decode a given fixed-length vector representation back into a variable-length sequence.

[- Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation, 2014.](https://arxiv.org/abs/1406.1078)

* [Encoder-Decoder Recurrent Neural Network Models for Neural Machine Translation](https://machinelearningmastery.com/encoder-decoder-recurrent-neural-network-models-neural-machine-translation/)
* [Understanding Encoder-Decoder Sequence to Sequence Model](https://towardsdatascience.com/understanding-encoder-decoder-sequence-to-sequence-model-679e04af4346)
	* The power of this model lies in the fact that it can map sequences of different lengths to each other.

## **Seq2seq Model with Attention**

* Attention significantly improves RNN Seq2seq model.
* with attention, RNN Seq2seq model does not forget source input.
* with attention, the decoder knows where to focus.
* downside: much more computation.

## **More Resources**
* [Tensorflow 2.0 / Keras - LSTM vs GRU Hidden States](https://tiewkh.github.io/blog/gru-hidden-state/)

* [LSTM is dead. Long Live Transformers!](https://www.youtube.com/watch?v=S27pHKBEp30)
	*  Leo Dirac (@leopd) talks about how LSTM models for Natural Language Processing (NLP) have been practically replaced by transformer-based models.  Basic background on NLP, and a brief history of supervised learning techniques on documents, from `bag of words`, through `vanilla RNNs` and `LSTM`.  Then there's a technical deep dive into how `Transformers` work with `multi-headed self-attention`, and `positional encoding`.  Includes sample code for applying these ideas to real-world projects.
* [Attention Is All You Need - Paper](https://arxiv.org/abs/1706.03762)

## **Transformers**

***Positional Encoding***
* [Visual Guide to Transformer Neural Networks - (Episode 1) Position Embeddings](https://www.youtube.com/watch?v=dichIcUZfOw)

![Position Embeddings](./img/position_embeddings.png)


***Scaled Dot Product Attention & Multi Head Attention***
* [Visual Guide to Transformer Neural Networks - (Episode 2) Multi-Head & Self-Attention](https://www.youtube.com/watch?v=mMa2PmYJlCo)
![](./img/scaled_dot_product_attention.png)
