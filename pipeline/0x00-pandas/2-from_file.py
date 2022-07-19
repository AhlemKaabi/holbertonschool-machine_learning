#!/usr/bin/env python3
""" From File - Pandas """

import pandas as pd


def from_file(filename, delimiter):
    """
    Method:
    -------
        Loads data from a file as a (pd.DataFrame)

    Parameters:
    -----------
        filename: the file to load from.
        delimiter: the column separator.

    Returns:
    --------
        The loaded pd.DataFrame.
    """
    return pd.read_csv(filename, delimiter=delimiter)
