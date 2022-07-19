#!/usr/bin/env python3
""" From Dictionary - Pandas """

import pandas as pd


Dictionary = {
    "First": [0.0, 0.5, 1.0, 1.5],
    "Second": ["one", "two", "three", "four"]
}

df = pd.DataFrame(Dictionary, index=list("ABCD"))
