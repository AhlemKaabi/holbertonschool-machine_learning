#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
conv_backward = __import__('2-conv_backward').conv_backward

if __name__ == "__main__":
    np.random.seed(0)
    lib = np.load('../data/MNIST.npz')
    X_train = lib['X_train']
    _, h, w = X_train.shape
    X_train_c = X_train[:10].reshape((-1, h, w, 1))
    # print(X_train_c.shape)

    W = np.random.randn(3, 3, 1, 2)
    b = np.random.randn(1, 1, 1, 2)
    # print(b)

    dZ = np.random.randn(10, h - 2, w - 2, 2)

    plt.imshow(X_train[0])
    plt.show()

    # dA, dW, db = conv_backward(dZ, X_train_c, W, b, padding="same")

    print(conv_backward(dZ, X_train_c, W, b, padding="valid"))

