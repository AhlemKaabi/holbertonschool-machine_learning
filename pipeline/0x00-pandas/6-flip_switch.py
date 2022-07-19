#!/usr/bin/env python3

from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# the rows and columns are transposed and
# the data is sorted in reverse chronological order:
df = df.T.iloc[:, ::-1]

print(df.tail(8))
