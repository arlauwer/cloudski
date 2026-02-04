import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from constants import *

def read_sed(name):
	dat = np.loadtxt(OUTLOCATION+name)

	# column 1: wavelength; E (keV)
	# column 2: total flux; lambda*F_lambda (W/m2)
	# column 3: transparent flux; lambda*F_lambda (W/m2)
	# column 4: direct primary flux; lambda*F_lambda (W/m2)
	# column 5: scattered primary flux; lambda*F_lambda (W/m2)
	# column 6: direct secondary flux; lambda*F_lambda (W/m2)
	# column 7: scattered secondary flux; lambda*F_lambda (W/m2)
	# column 8: transparent secondary flux; lambda*F_lambda (W/m2)

	wav = dat[:, 0]
	tot = dat[:, 1]
	tr1 = dat[:, 2]
	di1 = dat[:, 3]
	sc1 = dat[:, 4]
	di2 = dat[:, 5]
	sc2 = dat[:, 6]
	tr2 = dat[:, 7]

	return wav, tot, tr1, tr2, di1, di2, sc1, sc2

def plot_sed(ax, sed):
	wav, tot, tr1, tr2, di1, di2, sc1, sc2 = sed
	ax.plot(wav, tr2, color='blue', linestyle='--')
	ax.plot(wav, di2, color='red', linestyle='--')
	ax.plot(wav, sc2, color='green', linestyle='--')
	ax.plot(wav, tot, color='black')
	ax.plot(wav, tr1, color='blue')
	ax.plot(wav, di1, color='red')
	ax.plot(wav, sc1, color='green')
	
	ax.set_xlabel("Wavelength (keV)")
	ax.set_ylabel("Flux (W/m2)")
	ax.set_xscale("log")
	ax.set_yscale("log")

	# custom legend
	custom_lines = [Line2D([0], [0], color='black', lw=2),
                	Line2D([0], [0], color='blue', lw=2),
					Line2D([0], [0], color='red', lw=2),
					Line2D([0], [0], color='green', lw=2),
					Line2D([0], [0], color='black', lw=2, linestyle='-'),
					Line2D([0], [0], color='black', lw=2, linestyle='--')]
	
	ax.legend(custom_lines, ['Total', 'Transparent', 'Direct', 'Scattered', 'Primary', 'Secondary'])

fig = plt.figure(figsize=(16, 8))
ax1, ax2 = fig.subplots(1, 2, sharex=True, sharey=True)

sed = read_sed("AGN_edge_sed.dat")
ax1.set_title("Edge")
plot_sed(ax1, sed)

sed = read_sed("AGN_face_sed.dat")
ax2.set_title("Face")
plot_sed(ax2, sed)

plt.show()