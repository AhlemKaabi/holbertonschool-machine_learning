#!/usr/bin/env python3

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization

if __name__ == '__main__':
    # np.random.seed(11)
    # a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    # b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    # c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    # d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    # X = np.concatenate((a, b, c, d), axis=0)
    # np.random.shuffle(X)
    # pi, m, S = initialize(X, 4)
    # g, _ = expectation(X, pi, m, S)
    # pi, m, S = maximization(X, g)
    # print(pi)
    # print(m)
    # print(S)

    X = np.random.randn(100, 6)
    print(maximization(X, 'hello'))
    print(maximization(X, np.array([1, 2, 3, 4, 5])))
    print(maximization(X, np.array([[[1, 2, 3, 4, 5]]])))
    g = np.random.randn(5, 90)
    g = g / np.sum(g, axis=0, keepdims=True)
    print(maximization(X, g))
    print(maximization(X, np.random.randn(5, 100)))