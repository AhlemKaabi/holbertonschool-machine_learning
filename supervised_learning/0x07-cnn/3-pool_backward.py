#!/usr/bin/env python3
"""
    CNN - Pooling Back Prop
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='avg'):
    """
    Method:
        performs back propagation over a pooling layer of
        a neural network:

    Parameters:
       @dA is a numpy.ndarray of shape (m, h_new, w_new, c_new)
         containing the partial derivatives with respect to the
         output of the pooling layer
        - m is the number of examples
        - h_new is the height of the output
        - w_new is the width of the output
        - c is the number of channels

        @A_prev(numpy.ndarray), shape (m, h_prev, w_prev, c)
          containing the output of the previous layer
        - h_prev: the height of the previous layer
        - w_prev: the width of the previous layer

        @kernel_shape: tuple of (kh, kw)
          containing the size of the kernel for the pooling
        - kh: the kernel height
        - kw: the kernel width

        @stride: tuple of (sh, sw)
          containing the strides for the pooling
        - sh: the stride for the height
        - sw: the stride for the width

        @mode: indicates the type of pooling
        - max: indicates max pooling
        - avg: indicates average pooling

    Returns:
        the partial derivatives with respect to
        the previous layer (dA_prev)
    """
    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)
    # dA_prev =  np.zeros(A_prev.shape)

    for m in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    x = h * sh
                    y = w * sw
                    if mode == 'max':
                        a_prev_slice = A_prev[m, x:x+kh, y:y+kw, c]

                        mask = (a_prev_slice == np.max(a_prev_slice))
                        # 1 for the cell that contains the max 0 otherwise
                        dA_prev[m, x:x+kh, y:y+kw, c] += mask * dA[m, h, w, c]

                    if mode == 'avg':
                        avg_dA = dA[m, h, w, c]/kh/kw

                        dA_prev[m, x:x+kh,
                                y:y+kw, c] += np.ones((kh, kh)) * avg_dA
    return dA_prev
