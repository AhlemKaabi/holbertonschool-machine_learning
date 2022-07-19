#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE
df.drop('Weighted_Price', inplace=True, axis=1)
df["Volume_(BTC)"].fillna(0, inplace=True)
df["Volume_(Currency)"].fillna(0, inplace=True)

df["Close"] = df["Close"].fillna(method="ffill")


df['High'] = df["Close"]
df["Low"] = df["Close"]
df["Open"] = df["Close"]

print(df.head())
print(df.tail())
