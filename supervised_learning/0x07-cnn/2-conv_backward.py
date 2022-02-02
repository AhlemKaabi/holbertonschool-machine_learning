#!/usr/bin/env python3
"""
    CNN - Convolutional Back Prop
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Method:
         Performs forward propagation over a convolutional
           layer of a neural network:

    Parameters:
        @dZ(numpy.ndarray), shape (m, h_new, w_new, c_new) containing
          the partial derivatives with respect to the unactivated
          output of the convolutional layer
            - m: the number of examples
            - h_new: the height of the output
            - w_new: the width of the output
            - c_new: the number of channels in the output
        @A_prev(numpy.ndarray), shape (m, h_prev, w_prev, c_prev):
          containing the output of the previous layer
            - h_prev: the height of the previous layer
            - w_prev: the width of the previous layer
            - c_prev: the number of channels in the previous layer
        @W(numpy.ndarray), shape (kh, kw, c_prev, c_new):
          containing the kernels for the convolution
            - kh: the filter height
            - kw: the filter width
        @b(numpy.ndarray), shape (1, 1, 1, c_new):
              containing the biases applied to the convolution
        @padding: string that is either same or valid - type of padding used
        @stride: tuple of (sh, sw) containing the strides for the convolution
            - sh: the stride for the height
            - sw: the stride for the width

    Returns:
        the partial derivatives with respect to the previous
        layer (dA_prev), the kernels (dW), and the biases (db),
        respectively


    """
    m, h_new, w_new, c_new = dZ.shape

    _, h_prev, w_prev, _ = A_prev.shape

    kh, kw, _, _ = W.shape

    sh, sw = stride

    if padding == "valid":
        ph = 0
        pw = 0
    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    A_prev_padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant')

    # Initializing with the correct shapes
    dA_prev = np.zeros(A_prev.shape)

    dW = np.zeros(W.shape)

    # for c in range(c_new):
    #     for k_h in range(kh):
    #         for k_w in range(kw):
    #             sum_slices = 0
    #             for m in range(m):
    #                 dz_f = dZ[m, :, :, c]
    #                 A_prev_slice = A_prev[:,
    #                                       0 + k_h: h_prev - h_new + k_h,
    #                                    0 + k_w: w_prev - w_new + k_w, :]
    #                 sum_slices += np.tensordot(A_prev_slice, dz_f)
    #             dW[k_h, k_w, :, c] = sum_slices

    for m in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for f in range(c_new):
                    filter_k = W[:, :, :, f]
                    x = h * sh
                    y = w * sw
                    dz = dZ[m, h, w, f]
                    A_prev_slice = A_prev_padded[m, x:x + kh, y:y+kw, :]

                    dW[:, :, :, f] += A_prev_slice * dz

                    dA_prev[m, x:x + kh, y:y+kw, :] += dz * filter_k
    # we have to remove the padding part (ph = ph = 0) 'valid'
    if padding == 'same':
        dA_prev = dA_prev[:, ph:-ph, pw:-pw, :]

    return dA_prev, dW, db
