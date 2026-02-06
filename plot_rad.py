from constants import *
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from read_rad import *
from read_opac import *
from read_props import *


def slider_plot_rad(fig, RAD, OPAC, props):

    edges, widths, rad = RAD
    opac_wav, opac = OPAC

    ax_r = fig.add_subplot(121)
    ax_o = fig.add_subplot(122)
    fig.subplots_adjust(bottom=0.2)

    num_iter, num_cells, num_bins = rad.shape

    # initial plot
    bar = ax_r.bar(edges[:-1], rad[0, 0, :], width=widths, align='edge', edgecolor='black', linewidth=0.5)

    lin = ax_o.plot(opac_wav, opac[0, 0, :])[0]

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

    # MIN = rad.min()
    # MAX = rad.max()
    ax_r.set_xlabel("Energy (keV)")
    ax_r.set_ylabel("$\\lambda J_\\lambda $(W/m2/sr)")
    ax_r.set_xscale('log')
    ax_r.set_yscale('log')
    # ax_r.set_ylim(MIN, MAX)

    MIN = opac.min()
    MAX = opac.max()
    ax_o.set_xlabel("Energy (keV)")
    ax_o.set_ylabel("Opacity")
    ax_o.set_xscale('log')
    ax_o.set_yscale('log')
    ax_o.set_ylim(MIN, MAX)

    def update(val):
        iter = int(slider_iter.val)
        cell = int(slider_cell.val)

        # Update the heights of the bars
        new_heights = rad[iter, cell, :]
        for rect, h in zip(bar, new_heights):
            rect.set_height(h)

        # Update the line
        lin.set_ydata(opac[iter, cell, :])

        # Update the title
        prop = props[cell]
        c, x, z = int(prop[0]), prop[1], prop[3]
        fig.suptitle(f"Cell {cell:03d}={c:03d} at (x,z)=({x:.2f},{z:.2f}) pc")

        fig.canvas.draw_idle()

    slider_iter.on_changed(update)
    slider_cell.on_changed(update)

    return slider_iter, slider_cell


if __name__ == "__main__":

    # rad
    edges, widths = read_rad_wav()
    rads = read_rads()

    # opac
    opac_wav = read_opac_wav()
    opac = read_opacs()

    # props
    props = radial_props()

    print("Number of (iterations, cells, bins):", rads.shape)

    assert widths.shape[0] == rads.shape[2]  # Amount of bins
    assert props.shape[0] == rads.shape[1]  # Amount of cells

    fig = plt.figure(figsize=(16, 8))

    sliders = slider_plot_rad(fig, (edges, widths, rads), (opac_wav, opac), props)

    plt.show()
