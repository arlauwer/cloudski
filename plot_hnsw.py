import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from read_hnsw import *

def slider_plot2D(fig, points, xi, yi, to_name=to_name, log=True):
    print("Plotting 2D Slider plot with shape: ", points.shape)
    dim = points.shape[1]

    ax = fig.add_subplot(111)
    fig.subplots_adjust(bottom=0.2)

    # initial plot
    scat = ax.scatter(points[:, xi], points[:, yi], c='k', s=2)

    # sliders
    ax_slider_x = plt.axes([0.15, 0.05, 0.7, 0.05])
    ax_slider_y = plt.axes([0.15, 0.10, 0.7, 0.05])
    slider_x = Slider(
        ax=ax_slider_x,
        label='x-axis',
        valmin=0,
        valmax=dim - 1,
        valinit=xi,
        valstep=1
    )
    slider_y = Slider(
        ax=ax_slider_y,
        label='y-axis',
        valmin=0,
        valmax=dim - 1,
        valinit=yi,
        valstep=1
    )

    if log:
        nonzero = points[points != 0]
        MIN = nonzero.min()
        MAX = nonzero.max()
    else:
        MIN = points.min()
        MAX = points.max()

    ax.set_xlabel(to_name(xi))
    ax.set_ylabel(to_name(yi))
    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlim(MIN, MAX)
    ax.set_ylim(MIN, MAX)

    def update(val):
        xi = int(slider_x.val)
        yi = int(slider_y.val)
        # Get current X and new Y data
        x_data = points[:, xi]
        y_data = points[:, yi]
        
        # set_offsets expects an (N, 2) array
        scat.set_offsets(np.c_[x_data, y_data])
        
        # Update label and redraw
        ax.set_xlabel(to_name(xi))
        ax.set_ylabel(to_name(yi))
        fig.canvas.draw_idle()

    # Register the update function
    slider_x.on_changed(update)
    slider_y.on_changed(update)

    return slider_x, slider_y


if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf)
    points = read_hnsw_points()
    print("Number of points:", points.shape[0])
    print("Number of dimensions: 2 +", points.shape[1]-2)

    fig = plt.figure(figsize=(20, 10))

    sliders = slider_plot2D(fig, points, 0, 1)

    plt.show()