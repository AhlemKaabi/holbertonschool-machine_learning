#!/usr/bin/env python3
"""
    CNN - Convolutional Forward Prop
"""
import numpy as np


def conv_forward(A_prev, W, b,
                 activation, padding="same", stride=(1, 1)):
    """
    Method:
         Performs forward propagation over a convolutional
           layer of a neural network:

    Parameters:
        @A_prev(numpy.ndarray), shape (m, h_prev, w_prev, c_prev):
          containing the output of the previous layer
            - m: the number of examples
            - h_prev: the height of the previous layer
            - w_prev: the width of the previous layer
            - c_prev: the number of channels in the previous layer
        @W(numpy.ndarray), shape (kh, kw, c_prev, c_new):
          containing the kernels for the convolution
            - kh: the filter height
            - kw: the filter width
            - c_prev: the number of channels in the previous layer
            - c_new: the number of channels in the output
        @b(numpy.ndarray), shape (1, 1, 1, c_new):
              containing the biases applied to the convolution
        @activation: activation function applied to the convolution
        @padding: string that is either same or valid - type of padding used
        @stride: tuple of (sh, sw) containing the strides for the convolution
            - sh: the stride for the height
            - sw: the stride for the width

    Returns:
        the output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape

    # c_previs the depth of the kernel == nb of channels
    # c_new is the numbers of filters
    kh, kw, c_prev, c_new = W.shape

    sh, sw = stride

    if padding == "valid":
        ph = 0
        pw = 0
    elif padding == "same":
        ph = int(np.ceil((h_prev - 1) * sh + kh - h_prev) / 2)
        pw = int(np.ceil((w_prev - 1) * sw + kw - w_prev) / 2)

    A_h = int(((h_prev + 2 * ph - kh) / sh) + 1)
    A_w = int(((w_prev + 2 * pw - kw) / sw) + 1)

    # A is the outputof the convolutional layer
    A = np.zeros((m, A_h, A_w, c_new))

    # Add padding to the input image
    image_padded = np.pad(A_prev,
                          pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                          mode='constant')

    # Loop over every pixel of the output layer
    for k in range(c_new):
        filter_K = W[:, :, :, k]
        for i in range(A_h):
            for j in range(A_w):
                x = i * sh
                y = j * sw
                # element-wise multiplication of the kernel and the image
                img_slice = image_padded[:, x:x+kh, y:y+kw, :]
                A[:, i, j, k] = np.tensordot(img_slice, filter_K, axes=3)
    A = activation(A + b)
    return A
