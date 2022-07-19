#!/usr/bin/env python3

from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
# https://sparkbyexamples.com/pandas/pandas-drop-rows-with-nan-values-in-dataframe/
df = df.dropna(subset=['Close'])

print(df.head())
