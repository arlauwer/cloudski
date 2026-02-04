import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from read_hnsw import *
from plot_hnsw import slider_plot2D

def scale(points):
	# Apply log scaling to all columns except Z
	mask = np.ones(points.shape[1], dtype=bool)
	mask[1] = False
	scaled = points.copy()
	scaled[:, mask] = np.log10(points[:, mask])

	# Standardize the log-transformed data
	# PCA is very sensitive to variance, so we must center/scale after the log
	scaler = StandardScaler()
	scaled = scaler.fit_transform(scaled)
	return scaled

def perform_pca(points, N):
	scaled = scale(points)

	pca = PCA(n_components=N)
	proj = pca.fit_transform(scaled)

	return pca, proj

def plot_cov(ax, cov):
	ax.imshow(cov,cmap='hot', interpolation='nearest')

if __name__ == "__main__":
	np.set_printoptions(precision=2, linewidth=np.inf)
	
	points = read_hnsw_points()
	print("Number of points:", points.shape[0])
	print("Number of dimensions: 2 +", points.shape[1]-2)

	pca, proj = perform_pca(points, 2)

	# PCA info
	cov1 = np.cov(scale(points), rowvar=False)
	cov2 = pca.get_covariance()

	fig1 = plt.figure(figsize=(16, 8))
	# fig1.canvas.manager.window.wm_geometry("+0+0")

	sliders = slider_plot2D(fig1, proj, 0, 1, lambda i: i, log=False)

	fig2 = plt.figure(figsize=(16, 8))
	# fig1.canvas.manager.window.wm_geometry("+0+1000")

	ax1 = fig2.add_subplot(121)
	ax2 = fig2.add_subplot(122)
	plot_cov(ax1, cov1)
	plot_cov(ax2, cov2)

	plt.show()