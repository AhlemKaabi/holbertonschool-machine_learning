#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# The column Weighted_Price should be removed
df = df.drop(columns =['Weighted_Price'])
# Rename the column Timestamp to Date
df.rename(columns={'Timestamp': 'Date'}, inplace=True)
# Convert the timestamp values to date values
df['Date'] = pd.to_datetime(df.Date, unit='s')
# Index the data frame on Date
df = df.set_index('Date')

# Missing values in Close should be set to the previous row value
df["Close"] = df["Close"].fillna(method="ffill")
# Missing values in High, Low, Open should be set to the same row’s Close value
df['High'] = df["Close"]
df["Low"] = df["Close"]
df["Open"] = df["Close"]
# Missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
df["Volume_(BTC)"].fillna(0, inplace=True)
df["Volume_(Currency)"].fillna(0, inplace=True)


# Plot the data from 2017 and beyond at daily intervals and group the values
# of the same day such that:
    # High: max
    # Low: min
    # Open: mean
    # Close: mean
    # Volume(BTC): sum
    # Volume(Currency): sum

df_graph = pd.DataFrame()
# groupby day and take the max
df_graph['High'] = df['High'].resample('D').max()
# groupby day and take the min
df_graph['Low'] = df['Low'].resample('D').min()
# groupby day and take the sum
df_graph['Volume_(BTC)'] = df['Volume_(BTC)'].resample('D').sum()
df_graph['Volume_(Currency)'] = df['Volume_(Currency)'].resample('D').sum()
# groupby day and take the mean
df_graph["Open"] = df["Open"].resample('D').mean()
df_graph["Close"] = df["Close"].resample('D').mean()


df_graph = df_graph[df_graph.index >= np.datetime64('2017')]

# plt.figure(16, 8)
df_graph.plot(figsize=(16, 8))
# plt.figure(figsize=(16, 8))
plt.savefig("df_graph")
