import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open('pd.txt', 'r', encoding='utf-8') as f:
	lines = f.readlines()

x, y, z = [], [], []

for line in lines:
	seg = line.split()
	x.append(float(seg[0]))
	y.append(float(seg[1]))
	z.append(float(seg[2]))


fig=plt.figure(dpi=120)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, marker='.', s=0.1, c='y') 
plt.show()