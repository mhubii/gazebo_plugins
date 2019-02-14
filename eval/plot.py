import numpy as np
from mpl_toolkits.mplot3d import Axes3D # 3d plots
import matplotlib.pyplot as plt
from matplotlib import cm # colormap
from scipy import signal # convolution

from utils import disc, bin_data

# load data
uniform = np.genfromtxt("circular_random_sample.csv", delimiter=",")
path = np.genfromtxt("../data/vehicle_positions.csv", delimiter=",")
other = np.genfromtxt("../data/goal_obstacle_positions.csv", delimiter=",")

goal = other[:,0:3]
obstacle = other[:,3:6]

bin_size = 1/10

# bin data
#binned = bin_data(uniform, 1/100, 0, 0, 200, 200, True)
binned = bin_data(obstacle[:, 0:2], bin_size=bin_size, center_x=2, center_y=0, padding_x=20, padding_y=20, normalize=True) # now with read instead of fake data

# create circle
disc = disc(int(1/bin_size), int(1/bin_size))

# convolve
conv = signal.convolve2d(binned, disc, boundary="symm", mode="same")

# plot dots
plt.scatter(obstacle[:, 0], obstacle[:, 1])
plt.show()

# plot paths
plt.plot(path[:,0], path[:,1])
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
