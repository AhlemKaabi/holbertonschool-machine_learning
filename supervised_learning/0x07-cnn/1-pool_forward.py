#!/usr/bin/env python3
"""
    CNN - Pooling Forward Prop
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Method:
         Performs forward propagation over a pooling
         layer of a neural network

    Parameters:
        @A_prev(numpy.ndarray), shape (m, h_prev, w_prev, c_prev):
          containing the output of the previous layer
            - m: the number of examples
            - h_prev: the height of the previous layer
            - w_prev: the width of the previous layer
            - c_prev: the number of channels in the previous layer
        @kernel_shape(numpy.ndarray), shape (kh, kw):
          containing the size of the kernel for the pooling
            - kh: the kernel height
            - kw: the kernel width
        @stride: tuple of (sh, sw) containing the strides for the pooling
            - sh: the stride for the height
            - sw: the stride for the width
        @mode: indicates the type of pooling
            - max: indicates max pooling
            - avg: indicates average pooling

    Returns:
        the output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape

    kh, kw = kernel_shape

    sh, sw = stride

    A_h = int(((h_prev - kh) / sh) + 1)
    A_w = int(((w_prev - kw) / sw) + 1)

    A = np.zeros((m, A_h, A_w, c_prev))

    # Loop over every pixel of the output
    for i in range(A_h):
        for j in range(A_w):
            x = i * sh
            y = j * sw
            img_slice = A_prev[:, x:x+kh, y:y+kw, :]
            if mode == 'max':
                A[:, i, j, :] = np.max(img_slice, axis=(1, 2))
            if mode == 'avg':
                A[:, i, j, :] = np.average(img_slice, axis=(1, 2))
    return A
