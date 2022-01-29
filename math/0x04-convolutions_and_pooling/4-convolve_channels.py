#!/usr/bin/env python3
"""
    Convolution with Channels
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Method:
        performs a convolution on images with channels.

    @images(numpy.ndarray), shape (m, h, w, c)
        containing multiple grayscale images.
        - m: the number of images
        - h: the height in pixels of the images
        - w: the width in pixels of the images
        - c: the number of channels in the image
    @kernel(numpy.ndarray), shape (kh, kw, c)
        containing the kernel for the convolution
        - kh: the height of the kernel
        - kw: the width of the kernel
    @padding: tuple of (ph, pw)
        - ph: the padding for the height of the image
        - pw: the padding for the width of the image
     @stride: tuple of (sh, sw)
        - sh: the stride for the height of the image
        - sw: the stride for the width of the image

    Returns:
        a numpy.ndarray containing the convolved images.
    """
    m, input_h, input_w, c = images.shape
    kernel_h, kernel_w, kernel_c = kernel.shape
    sh, sw = stride

    if padding == 'valid':
        ph = 0
        pw = 0
    elif padding == "same":
        ph = int(np.ceil((input_h - 1) * sh + kernel_h - input_h) / 2)
        pw = int(np.ceil((input_w - 1) * sw + kernel_w - input_w) / 2)
    else:
        ph = padding[0]
        pw = padding[1]

    output_height = int(((input_h + 2 * ph - kernel_h) / sh) + 1)
    output_width = int(((input_w + 2 * pw - kernel_w) / sw) + 1)

    # Same convolution output
    output = np.zeros((m, output_height, output_width))

    # Add zero padding to the input image
    image_padded = np.pad(images,
                          pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                          mode='constant')

    # Loop over every pixel of the output
    for i in range(output_height):
        for j in range(output_width):
            x = i * sh
            y = j * sw
            # element-wise multiplication of the kernel and the image
            img_slice = image_padded[:, x:x+kernel_h, y:y+kernel_w]
            output[:, i, j] = np.tensordot(img_slice, kernel, axes=3)
    return output
