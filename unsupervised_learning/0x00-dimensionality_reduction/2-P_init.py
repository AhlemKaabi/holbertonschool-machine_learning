#!/usr/bin/env python3
"""
	Initialize t-SNE!!
"""
import numpy as np


def P_init(X, perplexity):
	"""
	Method:
 		Initializes all variables required to calculate
   		the P affinities in t-SNE.

    Args:
		X[numpy.ndarray] shape (n, d):
        - n: the number of data points
        - d: the number of dimensions in each point
		perplexity: the perplexity that
  		all Gaussian distributions should have

    Returns: (D, P, betas, H)
		D[numpy.ndarray] (n, n):
			- calculates the squared pairwise distance between
   			two data points, The diagonal of D should be 0s.
		P[numpy.ndarray] (n, n):
			- initialized to all 0‘s that will contain the
   			P affinities
		betas[numpy.ndarray] (n, 1):
			- initialized to all 1’s that will contain all of
   			the beta values: (b_i = 1 / 2 * sigma_i^2)
		H: -the Shannon entropy for perplexity, with a base
  		of 2
	"""
	n, d = X.shape

	P = np.zeros((n, n))
	betas = np.ones((n, 1))
	# The Shannon entropy
	H =  np.log2(perplexity)
	# 
	D =