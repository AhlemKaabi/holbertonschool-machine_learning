#!/usr/bin/env python3
""" From Numpy - Pandas """

import pandas as pd


def from_numpy(array):
    """
    Methods:
    --------
        Creates a (pd.DataFrame) from a (np.ndarray)

    Parameters:
    -----------
        array(np.ndarray): from which you should create the (pd.DataFrame)

    Returns:
    --------
        The newly created pd.DataFrame
    """
    # The columns of the pd.DataFrame should be labeled in alphabetical order
    # and capitalized. There will not be more than 26 columns.
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    return pd.DataFrame(
        array,
        columns=alphabet[:array.shape[1]]
    )
