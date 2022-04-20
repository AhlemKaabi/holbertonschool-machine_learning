# **Autoencoders**

## **Learning Objectives**

* What is an autoencoder?
* What is latent space?
* What is a bottleneck?
* What is a sparse autoencoder?
* What is a convolutional autoencoder?
* What is a generative model?
* What is a variational autoencoder?
* What is the Kullback-Leibler divergence?



An `autoencoder` is a type of artificial neural network used to learn efficient codings of `unlabeled` data (unsupervised learning). The encoding is validated and refined by attempting to regenerate the input from the encoding. The autoencoder learns a representation (encoding) for a set of data, typically for dimensionality reduction, by training the network to ignore insignificant data (“noise”).

![autoencoders](./img/autoencoders.png)


Types of autoencoder:([Deep inside: Autoencoders](https://towardsdatascience.com/deep-inside-autoencoders-7e41f319999f))

In this article, the four following types of autoencoders will be described:

* Vanilla autoencoder
* Multilayer autoencoder
* Convolutional autoencoder
* Regularized autoencoder

Applications of autoencoders include:

* Anomaly detection
* Data denoising (ex. images, audio)
* Image inpainting
* Information retrieval


