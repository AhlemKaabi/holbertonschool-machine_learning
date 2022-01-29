#!/usr/bin/env python3
"""
    Valid Convolution
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Method:
        performs a valid convolution on grayscale images.

    @images(numpy.ndarray), shape (m, h, w)
        containing multiple grayscale images.
        - m: the number of images
        - h: the height in pixels of the images
        - w: the width in pixels of the images
    @kernel(numpy.ndarray), shape (kh, kw)
        containing the kernel for the convolution
        - kh: the height of the kernel
        - kw: the width of the kernel

    Returns:
        a numpy.ndarray containing the convolved images
    """
    # only allowed to use two for loops;
    # any other loops of any kind are not allowed

    # input_width and input_height
    m, input_h, input_w = images.shape
    # flip the kernal matrix
    # kernel = np.transpose(kernel) (already!)
    # kernel_width and kernel_height
    kernel_h, kernel_w = kernel.shape

    output_height = input_h - kernel_h + 1
    output_width = input_w - kernel_w + 1

    # valid convolution output
    output = np.zeros((m, output_height, output_width))

    # Loop over every pixel of the output
    for x in range(output_width):
        for y in range(output_height):
            # element-wise multiplication of the kernel and the image
            img_slice = images[:, y:y+kernel_h, x:x+kernel_w]
            # https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html
            # output[:, y, x] = np.multiply(kernel, img_slice).sum(axis=(1, 2))
            # tensors order! np.tensordot(a, b, axes=N) axes=2 default
            # the last N dimensions of a and the first
            # N dimensions of b are summed over.
            output[:, y, x] = np.tensordot(img_slice, kernel)
    return output
