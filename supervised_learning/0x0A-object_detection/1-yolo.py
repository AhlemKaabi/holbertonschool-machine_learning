#!/usr/bin/env python3
"""
    Class Yolo that uses
    the Yolo v3 algorithm to perform object detection.
"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Method:
            class constructor

        Args:
            * model_path: Path to where a Darknet Keras model is stored

            * classes_path: Path to where the list of class names used
            for the Darknet model, listed in order of index, can be found

            * class_t(float): representing the box score threshold for the
            initial filtering step

            * nms_t(float): representing the IOU threshold for
            non-max suppression

            * anchors(numpy.ndarray), shape (outputs, anchor_boxes, 2)
              containing all of the anchor boxes:
              - outputs: the number of outputs (predictions)
                made by the Darknet model
              - anchor_boxes: number of anchor boxes used for each prediction
              - 2 => [anchor_box_width, anchor_box_height]
        """
        # the Darknet Keras model
        self.model = K.models.load_model(model_path)
        # a list of the class names for the model
        with open(classes_path) as f:
            class_names_list = f.readlines()
        self.class_names = class_names_list
        # the box score threshold for the initial filtering step
        self.class_t = class_t
        # the IOU threshold for non-max suppression
        self.nms_t = nms_t
        # the anchor boxes
        self.anchors = anchors

    def sigmoid(self, x):
        """
        Method:
            compute the sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Method:
            Process outputs of the Darknet model
        Args:
            * outputs:
            predictions from the Darknet model for a single image
              ->(type: list of numpy.ndarrays)
              -> numpy.ndarrays shape
              (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
              - grid_height =>  the height of the grid used for the output
              - grid_width =>  the width of the grid used for the output
              - anchor_boxes => the number of anchor boxes used
              - 4 => (t_x, t_y, t_w, t_h)
              - 1 => box_confidence
              - classes => class probabilities for all classes

            * image_size:
            the image's original size [image_height, image_width]
            ->(type: numpy.ndarray)
        Returns:
            tuple of (boxes, box_confidences, box_class_probs)

            * boxes:
            -> type: list of numpy.ndarrays
            -> shape (grid_height, grid_width, anchor_boxes, 4)
            containing the processed boundary boxes for each output
            - 4 => (x1, y1, x2, y2)
            - (x1, y1, x2, y2) should represent the boundary box
            relative to original image

            * box_confidences:
            -> type: list of numpy.ndarrays
            -> shape (grid_height, grid_width, anchor_boxes, 1)
            containing the box confidences for each output, respectively

            * box_class_probs:
            -> type: list of numpy.ndarrays
            -> shape (grid_height, grid_width, anchor_boxes, classes)
            containing the box’s class probabilities for each output
        """

        boxes = []
        box_confidences = []
        box_class_probs = []

        # processing each output per scale (3 scales)
        # each scale (13x13), (26x26), (52,52)
        for idx, output in enumerate(outputs):
            t_xy = output[..., :2]
            t_wh = output[..., 2:4]
            # box_confidences is confidence score indicates
            # the likelihood that the cell contains an object
            box_confidence = np.expand_dims(self.sigmoid(output[..., 4]),
                                            axis=-1)
            box_class_prob = self.sigmoid(output[..., 5:])

            # get anchor of the current output element
            anchors = self.anchors[idx]
            # get grid width and height
            g_w, g_h = output.shape[:2]

            grid = np.tile(np.indices((g_w, g_h)).T,
                           3).reshape((g_h, g_w) + anchors.shape)

            # computing b_xy, b_wh
            b_xy = self.sigmoid(t_xy) + grid

            b_wh = anchors * np.exp(t_wh)

            # normalizing b_wh and b_xy
            # b_wh; divide by the model's input shape (416x416)
            # b_xy; divide by the grid size (1 scale 13x13)

            b_wh /= self.model.inputs[0].shape_as_list()[1:3]
            b_xy /= [g_w, g_h]

            b_xy1 = b_xy - (b_wh / 2)
            b_xy2 = b_xy - (b_wh / 2)

            box = np.concatenate((b_xy1, b_xy2), axis=-1)

            # multiply by the original image that I fet to my neural network
            # because the network was trained(expecting) images shape (416x416)

            box = box * np.tile(np.flip(image_size, axis=0), 2)

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs
