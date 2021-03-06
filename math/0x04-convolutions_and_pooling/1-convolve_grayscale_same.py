#!/usr/bin/env python3
"""
    Same Convolution
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Method:
        performs a same convolution on grayscale images.

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

    output_height = input_h
    output_width = input_w

    # Calculate the number of zeros which
    # are needed to add as padding (height & width)
    ph = int(np.ceil((kernel_h - 1) / 2))
    pw = int(np.ceil((kernel_w - 1) / 2))

    # Same convolution output
    output = np.zeros((m, output_height, output_width))

    # Add zero padding to the input image
    image_padded = np.pad(images,
                          pad_width=((0, 0), (ph, ph), (pw, pw)),
                          mode='constant')

    # Loop over every pixel of the output
    for x in range(output_width):
        for y in range(output_height):
            # element-wise multiplication of the kernel and the image
            img_slice = image_padded[:, y:y+kernel_h, x:x+kernel_w]
            output[:, y, x] = np.tensordot(img_slice, kernel)
    return output
