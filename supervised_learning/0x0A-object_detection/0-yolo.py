#!/usr/bin/env python3
"""
    Class Yolo that uses
    the Yolo v3 algorithm to perform object detection.
"""
import tensorflow.keras as K


class Yolo:
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Method:
            class constructor

        Args:
            model_path: Path to where a Darknet Keras model is stored

            classes_path: Path to where the list of class names used
            for the Darknet model, listed in order of index, can be found

            class_t(float): representing the box score threshold for the
            initial filtering step

            nms_t(float): representing the IOU threshold for
            non-max suppression

            anchors(numpy.ndarray), shape (outputs, anchor_boxes, 2)
              containing all of the anchor boxes:
              - outputs is the number of outputs (predictions)
              made by the Darknet model
              - anchor_boxes is the number of anchor boxes
              used for each prediction
                2 => [anchor_box_width, anchor_box_height]
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
