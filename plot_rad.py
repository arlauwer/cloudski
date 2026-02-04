from constants import *
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def read_wav(filename):
	# left and right edges of the bins
	widths, lborder, rborder = np.loadtxt(OUTLOCATION+filename, usecols=(1, 2, 3)).T
	num_wav = lborder.size
	edges = np.zeros(num_wav+1)
	edges[0] = lborder[0]
	edges[1:] = rborder
	return edges, widths

def read_rad(filename):
	dat = np.loadtxt(OUTLOCATION+filename)
	num_cells = dat.shape[0]
	if np.any(dat[:, 0] != np.arange(num_cells)):
		print("Error: Cell indices are not in the correct order")
		return

	print("Read", filename)
	return dat[:, 1:]


def read_rads():
	rad_files = glob.glob(OUTLOCATION+"AGN_rad-grid*_J.dat")

	num_iter = len(rad_files)

	rads = []
	for i in range(num_iter - 1):
		filename = "AGN_rad-grid"+f"{i+1:03d}"+"_J.dat"
		rads.append(read_rad(filename))
	rads.append(read_rad("AGN_rad-grid_J.dat"))

	return np.array(rads)

def slider_plot_rad(fig, edges, widths, rads):
	ax = fig.add_subplot(111)
	fig.subplots_adjust(bottom=0.2)

	# initial plot
	bar = ax.bar(edges[:-1], rads[0, 0, :], width=widths, align='edge', edgecolor='black', linewidth=0.5)

	# sliders
	ax_slider_iter = plt.axes([0.15, 0.05, 0.7, 0.05])
	ax_slider_cell = plt.axes([0.15, 0.10, 0.7, 0.05])
	slider_iter = Slider(
		ax=ax_slider_iter,
		label='iteration',
		valmin=0,
		valmax=num_iter-1,
		valinit=0,
		valstep=1
	)
	slider_cell = Slider(
		ax=ax_slider_cell,
		label='cell',
		valmin=0,
		valmax=num_cells-1,
		valinit=0,
		valstep=1
	)

	MIN = np.mean(rads) * 10e-4
	MAX = rads.max()

	ax.set_xlabel("Energy (keV)")
	ax.set_ylabel("$\\lambda J_\\lambda $(W/m2/sr)")
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlim(MIN, MAX)
	ax.set_ylim(MIN, MAX)

	def update(val):
		iter = int(slider_iter.val)
		cell = int(slider_cell.val)

		new_heights = rads[iter, cell, :]

		for rect, h in zip(bar, new_heights):
			rect.set_height(h)

		fig.canvas.draw_idle()

	slider_iter.on_changed(update)
	slider_cell.on_changed(update)

	return slider_iter, slider_cell



edges, widths = read_wav("AGN_rad-wav_wavelengths.dat")
rads = read_rads()

num_iter = rads.shape[0]
num_cells = rads.shape[1]
num_bins = rads.shape[2]
print("Number of (iterations, cells, bins):", rads.shape)

fig = plt.figure(figsize=(16, 8))

sliders = slider_plot_rad(fig, edges, widths, rads)

plt.show()