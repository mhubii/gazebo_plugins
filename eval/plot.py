import numpy as np
from mpl_toolkits.mplot3d import Axes3D # 3d plots
import matplotlib.pyplot as plt
from matplotlib import cm # colormap
from scipy import signal # convolution

from utils import disc, bin_data

def plot_conv():
	# load data
	uniform = np.genfromtxt("build/circular_random_sample.csv", delimiter=",")
	path = np.genfromtxt("data/vehicle_positions.csv", delimiter=",")
	other = np.genfromtxt("data/goal_obstacle_positions.csv", delimiter=",")
	mean = np.genfromtxt("data/mean_loss_score.csv", delimiter=",")

	goal = other[:,0:3]
	obstacle = other[:,3:6]

	bin_size = 1/10

	# bin data
	#binned = bin_data(uniform, 1/100, 0, 0, 200, 200, True)
	binned = bin_data(obstacle[:, 0:2], bin_size=bin_size, center_x=2, center_y=0, padding_x=20, padding_y=20, normalize=True) # now with read instead of fake data

	# create circle
	circ = disc(int(1/bin_size), int(1/bin_size))

	# convolve
	conv = signal.convolve2d(binned, circ, boundary="symm", mode="same")

	# plot obstacle and goal dots
	ax = fig.add_subplot(221)
	plt.scatter(obstacle[:, 0], obstacle[:, 1], c='r', label='obstacle')
	plt.scatter(goal[:,0], goal[:,1], c='g', label='goal')
	plt.title("Obstacle and Goal Positions")
	plt.legend()
	plt.xlabel("x")
	plt.ylabel("y")

	# plot paths
	ax = fig.add_subplot(222)
	plt.plot(path[:,0], path[:,1], zorder=1, label="paths")
	plt.scatter(obstacle[:, 0], obstacle[:, 1], c='r', zorder=2, label="obstacle")
	plt.scatter(goal[:, 0], goal[:, 1], c='g', zorder=2, label="goal")
	plt.title("Paths")
	plt.xlabel("x")
	plt.ylabel("y")
	plt.legend()

	# plot convolution
	ax = fig.add_subplot(223, projection='3d')

	# Plot the surface
	x = np.arange(0,conv.shape[1],1)
	y = np.arange(0,conv.shape[0],1)
	xx, yy = np.meshgrid(x, y)
	ax.plot_surface(xx, yy, conv, cmap=cm.coolwarm)
	plt.title("Effective Obstacle")

	# plot loss
	ax = fig.add_subplot(224)
	plt.plot(mean[:,0], mean[:,2])
	plt.title("Reward")
	plt.xlabel("Episode")
	plt.ylabel("a.e.")

	plt.tight_layout()

if __name__ == "__main__":
	fig = plt.figure()

	plot_conv()

	#plt.show()
	plt.savefig("result.pdf")
