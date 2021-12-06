#!/usr/bin/env python3
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

labels =['Farrah', 'Fred', 'Felicia']

# [[ 3 14 15]
#  [ 6 16  9]
#  [ 8  4  7]
#  [16 16  7]]
A = fruit[0]
B = fruit[1]
O = fruit[2]
P = fruit[3]
fig, ax = plt.subplots()

ax.bar(labels, A, label='apples', color='r', width=0.5)
ax.bar(labels, B, label='bananas', bottom=A, color='yellow', width=0.5)
ax.bar(labels, O, label='oranges', bottom=A+B, color='#ff8000', width=0.5)
ax.bar(labels, P, label='peaches', bottom=A+B+O, color='#ffe5b4', width=0.5)

ax.set_ylabel('Quantity of Fruit')
ax.set_ylim(0, 80)
ax.set_ylabel('Number of Fruit per Person')
ax.legend()
plt.show()
# your code here