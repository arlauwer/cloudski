import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Slider
from cloudy_manager import simgen, Bins
import numpy as np


def plot_single(ax, stab, xquant, fixed, **kwargs):
    yquant = stab['quantityNames'][0]
    x = stab[xquant].value
    y = stab[yquant].value

    fixed = fixed or {}

    slicer = []
    for axisName in stab['axisNames']:
        if axisName == xquant:
            slicer.append(slice(None))
        elif axisName in fixed:
            slicer.append(fixed[axisName])
        else:
            raise ValueError(f"Axis {axisName} not fixed")
    yi = y[tuple(slicer)]

    ax.plot(x, yi, **kwargs)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(f"{xquant} [{stab['axisUnits'][stab['axisNames'].index(xquant)]}]")
    ax.set_ylabel(f"{yquant} [{stab['quantityUnits'][stab['quantityNames'].index(yquant)]}]")


def plot_sweep(ax, stab, xquant, coloraxis, fixed, sm=None, **kwargs):
    yquant = stab['quantityNames'][0]
    x = stab[xquant].value
    y = stab[yquant].value

    cvals = stab[coloraxis].value
    cmap = plt.cm.get_cmap("rainbow_r")
    norm = mpl.colors.LogNorm(vmin=cvals.min(), vmax=cvals.max())

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
                raise ValueError(f"Axis {a} not fixed or swept")
        yi = y[tuple(slicer)]
        ax.plot(x, yi, color=cmap(norm(c)), **kwargs)

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


def plot_with_rad(fig, stab1, stab2, edges, params, xquant, fixed, **kwargs):
    # --- prepare data and axes ---
    yquant1 = stab1['quantityNames'][0]
    yquant2 = stab2['quantityNames'][0]
    x1 = stab1[xquant].value
    x2 = stab2[xquant].value
    y1 = stab1[yquant1].value
    y2 = stab2[yquant2].value
    axis_names = list(stab1['axisNames'])
    if axis_names != list(stab2['axisNames']):
        raise ValueError("stab1 and stab2 must have the same axisNames")

    # create axes (kept positions from your original layout)
    ax1 = fig.add_axes([0.10, 0.10, 0.35, 0.60])
    ax2 = fig.add_axes([0.55, 0.10, 0.35, 0.60])
    ax3 = fig.add_axes([0.10, 0.75, 0.80, 0.20])

    # --- bins handling ---
    bin_keys = [k for k in params.keys() if k.startswith('bin')]
    num_bins = len(bin_keys)
    if any(k != f"bin{idx}" for idx, k in enumerate(bin_keys)):
        raise ValueError(f"Expected bin keys named bin0..binN; got {bin_keys}")
    if edges.size != num_bins + 1:
        raise ValueError(f"edges length must be num_bins+1 ({num_bins+1}), got {edges.size}")

    # allowed arrays for each bin (as numpy arrays)
    bin_allowed = [np.asarray(params[k]) for k in bin_keys]

    # initial value for each bin: choose middle element (more sensible than first)
    current_vals = [arr[0] for arr in bin_allowed]

    # centers for marker placement (geometric mean for log spacing)
    centers = np.sqrt(edges[:-1] * edges[1:])

    # --- helper to find index for a chosen bin value inside params[binKey] ---
    def find_value_index(arr, val):
        arr = np.asarray(arr)
        # try exact/isclose match first
        idxs = np.where(np.isclose(arr, val, rtol=1e-8, atol=1e-12))[0]
        if idxs.size:
            return int(idxs[0])
        # fallback to nearest in log space if positive else linear nearest
        if np.all(arr > 0) and val > 0:
            return int(np.argmin(np.abs(np.log10(arr) - np.log10(val))))
        return int(np.argmin(np.abs(arr - val)))

    # --- initial slicer and draw ax1/ax2 ---
    def make_slicer():
        sl = []
        for axisName in axis_names:
            if axisName == xquant:
                sl.append(slice(None))
            elif axisName in fixed:
                sl.append(fixed[axisName])
            elif axisName in bin_keys:
                idx = find_value_index(params[axisName], current_vals[bin_keys.index(axisName)])
                sl.append(idx)
            else:
                raise ValueError(f"Axis {axisName} not fixed or not a bin: {axisName}")
        return tuple(sl)

    slicer = make_slicer()
    yline1 = np.asarray(y1[slicer])
    yline2 = np.asarray(y2[slicer])

    # left plot
    ax1_line, = ax1.plot(x1, yline1, lw=1.5)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title(f"{yquant1}")
    ax1.set_xlabel(f"{xquant} [{stab1['axisUnits'][stab1['axisNames'].index(xquant)]}]")
    ax1.set_ylabel(f"{yquant1} [{stab1['quantityUnits'][stab1['quantityNames'].index(yquant1)]}]")

    # right plot
    ax2_line, = ax2.plot(x2, yline2, lw=1.5)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title(f"{yquant2}")
    ax2.set_xlabel(f"{xquant} [{stab2['axisUnits'][stab2['axisNames'].index(xquant)]}]")
    ax2.set_ylabel(f"{yquant2} [{stab2['quantityUnits'][stab2['quantityNames'].index(yquant2)]}]")

    # --- top histogram-like control (ax3) ---
    # build step arrays for plotting a histogram-like stepped line
    def step_arrays(edges_arr, vals):
        # edges length m, vals length m-1 => x_step length 2*(m-1), y_step same
        x_step = np.repeat(edges_arr, 2)[1:-1]
        y_step = np.repeat(vals, 2)
        return x_step, y_step

    vals = np.array(current_vals, dtype=float)
    x_step, y_step = step_arrays(edges, vals)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    hist_line, = ax3.plot(x_step, y_step, lw=2, solid_capstyle='butt')
    markers = ax3.scatter(centers, vals, s=80, zorder=5, picker=True)
    ax3.set_xlim(edges[0], edges[-1])
    # set reasonable y limits from allowed sets
    all_allowed = np.concatenate(bin_allowed)
    pos_allowed = all_allowed[all_allowed > 0]
    if pos_allowed.size:
        ax3.set_ylim(pos_allowed.min()*0.8, pos_allowed.max()*1.2)
    ax3.set_title("Radiation field bins")
    ax3.set_xlabel("Bin edges [eV]")
    ax3.set_ylabel("$J_\\lambda \\Delta \\lambda$ [W/m2]")

    # --- interactive handlers ---
    dragging = {'idx': None}

    def update_hist_plot():
        nonlocal vals, x_step, y_step
        vals = np.array(current_vals, dtype=float)
        x_step, y_step = step_arrays(edges, vals)
        hist_line.set_data(x_step, y_step)
        markers.set_offsets(np.c_[centers, vals])
        fig.canvas.draw_idle()

    def update_main_plots():
        try:
            new_slicer = make_slicer()
            new_y1 = np.asarray(y1[new_slicer])
            new_y2 = np.asarray(y2[new_slicer])
        except Exception as e:
            print("update_main_plots: failed to index y with slicer:", e)
            return

        ax1_line.set_ydata(new_y1)
        ax2_line.set_ydata(new_y2)

        pos1 = new_y1[new_y1 > 0]
        if pos1.size:
            ymin, ymax = pos1.min()*0.8, pos1.max()*1.2
            cur_ymin, cur_ymax = ax1.get_ylim()
            ax1.set_ylim(min(ymin, cur_ymin), max(ymax, cur_ymax))
        
        pos2 = new_y2[new_y2 > 0]
        if pos2.size:
            ymin, ymax = pos2.min()*0.8, pos2.max()*1.2
            cur_ymin, cur_ymax = ax2.get_ylim()
            ax2.set_ylim(min(ymin, cur_ymin), max(ymax, cur_ymax))

        fig.canvas.draw_idle()


    def on_press(event):
        if event.inaxes != ax3 or event.xdata is None or event.ydata is None:
            return
        # find nearest center in log-x space
        if event.xdata <= 0:
            return
        log_click_x = np.log10(event.xdata)
        log_centers = np.log10(centers)
        dx = np.abs(log_centers - log_click_x)
        idx = int(np.argmin(dx))
        # pick only if reasonably close horizontally (0.5 decade threshold)
        if dx[idx] < 0.5:
            dragging['idx'] = idx

    def on_motion(event):
        if dragging['idx'] is None or event.inaxes != ax3 or event.ydata is None:
            return
        idx = dragging['idx']
        # choose nearest allowed value in log space (if positive)
        allowed = bin_allowed[idx]
        if event.ydata <= 0:
            # ignore negative or zero positions on log-scale
            return
        if np.all(allowed > 0):
            choice = allowed[np.argmin(np.abs(np.log10(allowed) - np.log10(event.ydata)))]
        else:
            choice = allowed[np.argmin(np.abs(allowed - event.ydata))]
        # update only if changed
        if choice != current_vals[idx]:
            current_vals[idx] = float(choice)
            update_hist_plot()
            update_main_plots()

    def on_release(event):
        dragging['idx'] = None

    # connect events
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)

    # final initial draw
    update_main_plots()
    fig.canvas.draw_idle()
    return ax1, ax2, ax3
