# **Holberton School Machine Learning**

### **Holberton School repository projects**
*This repository made in the framework of Holberton school projects to learn and practice machine learning conceptes and algorithms.*

## **First Section**
### **Machine Learning - Math**

* [0x00. Linear Algebra](./math/0x00-linear_algebra/)
* [0x01. Plotting](./math/0x01-plotting/)
* [0x02. Calculus](./math/0x02-calculus/)
* [0x03. Probability](./math/0x03-probability/)
* [0x04. Convolutions and Pooling](./math/0x04-convolutions_and_pooling/)
### **Machine Learning - Supervised Learning**

* 0x00. Classification - reading
* [0x01. Classification](./supervised_learning/0x01-classification/)
* [0x02. Tensorflow](./supervised_learning/0x02-tensorflow/)
* [0x03. Optimization](./supervised_learning/0x03-optimization/)
* 0x04. Error Analysis
* [0x05. Regularization](./supervised_learning/0x05-regularization/)
* [0x06. Keras](./supervised_learning/0x06-keras/)
* [0x07. Convolutional Neural Networks](./supervised_learning/0x07-cnn/)
* [0x08. Deep Convolutional Architectures](./supervised_learning/0x08-deep_cnns/)
* [0x09. Transfer Learning](./supervised_learning/0x09-transfer_learning/)
* [0x0A. Object Detection](./supervised_learning/0x0A-object_detection/)
* 0x0C. Neural Style Transfer


## **Second Section**

### **Machine Learning - Math**

* [0x05. Advanced Linear Algebra](./math/0x05-advanced_linear_algebra/)

* [0x06. Multivariate Probability](./math/0x06-multivariate_prob/)

* [0x07. Bayesian Probability](./math/0x07-bayesian_prob/)

### **Machine Learning - Unsupervised Learning**

* [0x00. Dimensionality Reduction](./unsupervised_learning/0x00-dimensionality_reduction/)

* [0x01. Clustering ](./unsupervised_learning/0x01-clustering/)

* [0x02. Hidden Markov Models](./unsupervised_learning/0x02-hmm/)

* [0x03. Hyperparameter Tuning](./unsupervised_learning/0x03-hyperparameter_tuning/)

* [0x04. Autoencoders](./unsupervised_learning/0x04-autoencoders/)

### **Machine Learning - Supervised Learning**

* [0x0D. RNNs](./supervised_learning/0x0D-RNNs/)
* [0x0E. Time Series Forecasting](./supervised_learning/0x0E-time_series/)
* [0x0F. Natural Language Processing - Word Embeddings](./supervised_learning/0x0F-word_embeddings/)
* [ 0x10. Natural Language Processing - Evaluation Metrics](./supervised_learning/0x10-nlp_metrics/)



## **Machine Learning Concepts and Terminology**

[](https://www.activestate.com/resources/quick-reads/what-is-a-keras-model/)

* `Accuracy.` Calculates the percentage of predicted values (yPred) that match actual values (yTrue).
* `Batch.` A set of N samples. Each sample in a batch is processed independently, in parallel with the other samples. Commonly referred to as a mini-batch.
* `Batch Size.` Number of samples processed through to the network at one time.
* `Convolutional Neural Network (CNN, or ConvNet).` Class of deep neural networks, commonly applied to analysis of visual imagery. Inspired by biological processes.
* `Epoch.` One single pass over the entire training set to the network. An arbitrary cutoff in training, defined as ‘one pass over the entire dataset’.
* `GPU.` Graphics Processing Unit. A TensorFlow processor platform that shows better flexibility and programmability for irregular computations, such as small batches. NVidia CUDA card requirement.
* `Gradient.` Slope of a function. Gradient measures the change in all weights with regard to the change in error.
* `Layer.` Instances of the layer() class are the basic building blocks in Keras neural networks. Consists of a tensor-in tensor-out computation function (the layer’s call method) and some state, held in TensorFlow variables.
* `Loss (L).` Measure of how far a model’s predictions are from its label. Metric that represents how good/bad a model is. Objective is to find a set of weights and biases that minimize loss. To determine loss, a model defines a loss function. Linear regression models typically use mean squared error while logistic regression models use Log Loss, for loss function.

Loss functions are available in the losses library. One of two required arguments for compiling a Keras model. To import the losses library, enter:
```
from keras import losses
```
Partial list of available loss functions:

		mean_squared_error
		mean_absolute _error
		hinge
		mean_absolute_percentage _error
		mean_squared_logarithmic_error
		Poisson
		binary_crossentropy
		categorical_crossentropy

***Holberton School repository projects - Machine learning Specialization***

*Project-based learning for Machine learning*