import numpy as np
from mpl_toolkits.mplot3d import Axes3D # 3d plots
import matplotlib.pyplot as plt
from matplotlib import cm # colormap
from scipy import signal # convolution

from utils import disc, bin_data

# load data
data = np.genfromtxt("circular_random_sample.csv", delimiter=",")

# bin data
binned = bin_data(data, 1/100, 200, 200, True)

# create circle
disc = disc(100, 100)

# convolve
conv = signal.convolve2d(binned, disc, boundary="symm", mode="same")

# plot dots
plt.scatter(data[:, 0], data[:, 1])
plt.plot()
plt.show()

# plot convolution
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
x = np.arange(0,conv.shape[1],1)
y = np.arange(0,conv.shape[0],1)
xx, yy = np.meshgrid(x, y)
ax.plot_surface(xx, yy, conv, cmap=cm.coolwarm)

plt.show()
