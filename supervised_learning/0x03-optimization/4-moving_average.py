#!/usr/bin/env python3
"""
  Moving Average
"""
import numpy as np


def moving_average(data, beta):
    """
    Method:
        Calculates the weighted moving average of a data set

    Args:
        @data: the list of data to calculate the moving
        average of
        @beta: the weight used for the moving average

    Returns:
        List containing the moving averages of data
    """
    # https://www.youtube.com/watch?v=k8fTYJPd3_I&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=20
    # Your moving average calculation should use bias correction
    weights = []
    Vt = 0
    for index, value in enumerate(data, start=1):
        Vt = beta * Vt + (1 - beta) * value
        # bias correction
        avg = Vt / (1 - beta**index)
        weights.append(avg)
    return weights
