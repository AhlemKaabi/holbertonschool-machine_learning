#!/usr/bin/env python3
"""
    Pooling
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Method:
        performs pooling on images.

    @images(numpy.ndarray), shape (m, h, w, c)
        containing multiple grayscale images.
        - m: the number of images
        - h: the height in pixels of the images
        - w: the width in pixels of the images
        - c: the number of channels in the image
    @kernel_shape:  tuple of (kh, kw)
        containing the kernel shape for the pooling
        - kh: the height of the kernel
        - kw: the width of the kernel
    @mode: indicates the type of pooling
        - max: indicates max pooling
        - avg: indicates average pooling

     @stride: tuple of (sh, sw)
        - sh: the stride for the height of the image
        - sw: the stride for the width of the image

    Returns:
        a numpy.ndarray containing the convolved images.
    """
    m, input_h, input_w, c = images.shape
    kernel_h, kernel_w = kernel_shape
    sh, sw = stride

    output_height = int(((input_h - kernel_h) / sh) + 1)
    output_width = int(((input_w - kernel_w) / sw) + 1)

    # Same convolution output
    output = np.zeros((m, output_height, output_width, c))

    # Loop over every pixel of the output
    for i in range(output_height):
        for j in range(output_width):
            x = i * sh
            y = j * sw
            # element-wise multiplication of the kernel and the image
            img_slice = images[:, x:x+kernel_h, y:y+kernel_w, :]
            if mode == 'max':
                output[:, i, j, :] = np.max(img_slice, axis=(1, 2))
            if mode == 'avg':
                output[:, i, j, :] = np.average(img_slice, axis=(1, 2))
    return output
