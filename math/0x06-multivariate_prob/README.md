# **Multivariate Probability**

### **Learning Objectives**

* Who is Carl Friedrich Gauss?
* What is a joint/multivariate distribution?
* What is a covariance?
* What is a correlation coefficient?
* What is a covariance matrix?
* What is a multivariate Gaussian distribution?


## Joint Probability Distributions

A joint probability distribution shows a probability distribution for two (or more) random variables. Instead of events being labeled A and B, the norm is to use X and Y.

f(x, y) = P(X = x, Y = y)

* Joint Probability Mass Function
If your variables are discrete (like in the above table example), their distribution can be described by a joint probability mass function (Joint PMF). Basically, if you have found all probabilities for all possible combinations of X and Y, then you have created a joint PMF.
* Joint Probability Density Function
If you have continuous variables, they can be described with a probability density function (PDF). Unlike the discrete variable example above, you can’t write out every combination of every variable because you would have infinite possibilities to write out (which is, of course, impossible). What you can do is create a formula; The formula that describes all possible combinations of X and Y is called a joint PDF.

[more informations](https://www.statisticshowto.com/joint-probability-distribution/)

## Multivariate Gaussian distributions

* **Gaussian distributions:**

	Gaussian distribution (also known as normal distribution) is a bell-shaped curve, and it is assumed that during any measurement values will follow a normal distribution with an equal number of measurements above and below the mean value.

	The normal distribution is the most common distribution function for independent, randomly generated variables. Its familiar bell-shaped curve is ubiquitous in statistical reports, from survey analysis and quality control to resource allocation.


* **Multivariate Gaussian distributions:**

	This is simply the extension to vector-valued random variables.
	In probability theory and statistics, the multivariate normal distribution, multivariate Gaussian distribution, or joint normal distribution is a generalization of the one-dimensional (univariate) normal distribution to higher dimensions.

![multi_dis](./img/multi_dis.png)

![multi_dis_graph](./img/multi_dis_graph.png)

### **0. Mean and Covariance:**

* Mean:
[numpy.mean](https://numpy.org/doc/stable/reference/generated/numpy.mean.html)

* Covariance matrix:


![covariance-matrix-formula](./img/covar.png)


*for Task 0, (file: 0-mean_cov.py) cov(X, X)!!*

```
cov_X_X = np.dot((X - mean).T, (X - mean)) / n - 1
```

### **1. Correlation :**
Are Covariance and Correlation The Same Thing?

Simply put, no.

While both covariance and correlation indicate whether variables are positively or inversely related to each other, they are not considered to be the same.

This is because correlation also informs about the degree to which the variables tend to move together.

Covariance is used to measure variables that have different units of measurement. By leveraging covariance, researchers are able to determine whether units are increasing or decreasing, but they are unable to solidify the degree to which the variables are moving together due to the fact that covariance does not use one standardized unit of measurement.

Correlation, on the other hand, standardizes the measure of interdependence between two variables and informs researchers as to how closely the two variables move together.

[more informations](https://www.alchemer.com/resources/blog/variance-covariance-correlation/)



[Converting between correlation and covariance matrices](https://blogs.sas.com/content/iml/2010/12/10/converting-between-correlation-and-covariance-matrices.html)

Correlation is just normalized Covariance refer to the formula below.

![image](./img/corr.png)


1. extract the variances from the diagonal elements of the covariance matrix
2. Then invert the matrix to form the diagonal matrix with diagonal elements that are the reciprocals of the standard deviations

*Variance: Variance is the average squared deviations from the mean, while standard deviation is the square root of this number.*



### **3. Multivariate Normal distribution Python Class**


class MultiNormal:

contains a public instance method `def pdf(self, x):` that calculates the PDF at a data point.

*Resources*
* [ Multivariate Normal Distribution](https://online.stat.psu.edu/stat505/book/export/html/636)
* [Multivariate Gaussian distributions](https://www.youtube.com/watch?v=eho8xH3E6mE)


***Holberton School Projects (Project-based learning Machine learning)***