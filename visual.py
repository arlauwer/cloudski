import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Slider


def plot_single(ax, stab, xquant, fixed=None, **kwargs):
    x = stab[xquant].value
    yquant = stab['quantityNames'][0]
    y = stab[yquant]

    fixed = fixed or {}

    slicer = []
    for axisName in stab['axisNames']:
        if axisName == xquant:
            slicer.append(slice(None))
        elif axisName in fixed:
            slicer.append(fixed[axisName])
        else:
            slicer.append(0)
    yi = y[tuple(slicer)]

    ax.plot(x, yi.value, **kwargs)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(f"{xquant} [{stab['axisUnits'][stab['axisNames'].index(xquant)]}]")
    ax.set_ylabel(f"{yquant} [{stab['quantityUnits'][stab['quantityNames'].index(yquant)]}]")


def plot_sweep(ax, stab, xquant, coloraxis, fixed=None, sm=None, **kwargs):
    x = stab[xquant].value
    yquant = stab['quantityNames'][0]
    y = stab[yquant]

    cvals = stab[coloraxis].value
    cmap = plt.cm.get_cmap("rainbow_r")
    norm = mpl.colors.LogNorm(vmin=cvals.min(), vmax=cvals.max())

    fixed = fixed or {}

    # for each point in the color axis, plot a line
    for i, c in enumerate(cvals):
        slicer = []
        for a in stab['axisNames']:
            if a == xquant:
                slicer.append(slice(None))
            elif a == coloraxis:
                slicer.append(i)
            elif a in fixed:
                slicer.append(fixed[a])
            else:
                slicer.append(0)
        yi = y[tuple(slicer)]
        ax.plot(x, yi.value, color=cmap(norm(c)), **kwargs)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(f"{xquant} [{stab['axisUnits'][stab['axisNames'].index(xquant)]}]")
    ax.set_ylabel(f"{yquant} [{stab['quantityUnits'][stab['quantityNames'].index(yquant)]}]")

    if sm is None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = ax.figure.colorbar(sm, ax=ax)
        cbar.set_label(f"{coloraxis} [{stab['axisUnits'][stab['axisNames'].index(coloraxis)]}]")
        return sm
    else:
        sm.set_norm(norm)
        return sm

# def interactive_bin_plot(stab, xquant, yquant, sweep_bin, params):
#     bin_axes = [k for k in params if k.startswith("bin") and k != sweep_bin]
#     n_sliders = len(bin_axes)

#     # leave space for sliders
#     slider_height = 0.04
#     bottom_margin = 0.05 + n_sliders * slider_height
#     fig, ax = plt.subplots(figsize=(10, 6))
#     plt.subplots_adjust(left=0.25, bottom=bottom_margin)

#     fixed = {b: 0 for b in bin_axes}
#     sm = plot_sweep(ax, stab, xquant, yquant, sweep_bin, fixed=fixed)

#     ax.set_xscale("log")
#     ax.set_yscale("log")
#     ax.set_ylim(1e-24, 1e-8)

#     sliders = {}
#     for j, b in enumerate(bin_axes):
#         ypos = 0.02 + j * slider_height
#         ax_slider = plt.axes([0.25, ypos, 0.65, 0.03])
#         slider = Slider(ax_slider, b, 0, len(params[b]) - 1, valinit=0, valstep=1)
#         sliders[b] = slider

#     def update(val):
#         fixed = {b: int(sliders[b].val) for b in bin_axes}
#         plot_sweep(ax, stab, xquant, yquant, sweep_bin, fixed=fixed, sm=sm)
#         ax.set_xscale("log")
#         ax.set_yscale("log")
#         ax.set_ylim(1e-24, 1e-8)
#         fig.canvas.draw_idle()

#     for s in sliders.values():
#         s.on_changed(update)

#     plt.show()
