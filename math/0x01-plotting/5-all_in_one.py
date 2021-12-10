#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(6, 5))
fig.suptitle('All in One')
y0 = np.arange(0, 11) ** 3

axs[0, 0].plot(y0, 'r')

# Scatter
mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

axs[0, 1].set_ylabel('Weight (lbs)', {'fontsize': 'x-small'})
axs[0, 1].set_xlabel('Height (in)', {'fontsize': 'x-small'})
axs[0, 1].set_title('Men\'s Height vs Weight', {'fontsize': 'x-small'})
axs[0, 1].scatter(x1, y1, c='m', marker='.')


# Change of scale
x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

axs[1, 0].set_ylabel('Fraction Remaining', {'fontsize': 'x-small'})
axs[1, 0].set_xlabel('Time (years)', {'fontsize': 'x-small'})
axs[1, 0].set_title('Exponential Decay of C-14', {'fontsize': 'x-small'})
axs[1, 0].set_yscale("log")
axs[1, 0].plot(x2, y2)


# code to plot x ↦ y1 and x ↦ y2 as line graphs
x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

axs[1, 1].set_ylabel('Fraction Remaining', {'fontsize': 'x-small'})
axs[1, 1].set_xlabel('Time (years)', {'fontsize': 'x-small'})
axs[1, 1].set_title('Exponential Decay of Radioactive Elements',
                    {'fontsize': 'x-small'})
axs[1, 1].plot(x3, y31, 'r--', label='C-14')
axs[1, 1].plot(x3, y32, 'g', label='Ra-226')
axs[1, 1].legend(loc='upper right')

# histogram of student scores
np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)
ax3 = plt.subplot2grid((3, 2), (2, 0), colspan=3)

ax3.hist(student_grades, bins=10, range=(0, 100), edgecolor='black')
ax3.set_ylim(0, 30)
ax3.set_ylabel('Number of Students', {'fontsize': 'x-small'})
ax3.set_xlabel('Grades', {'fontsize': 'x-small'})
ax3.set_title('Project A', {'fontsize': 'x-small'})

# figure

fig.tight_layout()
plt.show()
