import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from cloudski.cloudy.constants import *


def compare_cloudy_skirt(cloudy, skirt, run_indices=None, colors=None):
    if len(cloudy.runs) != len(skirt.runs):
        raise ValueError("Number of Cloudy and SKIRT runs do not match")

    if run_indices is None:
        run_indices = range(len(cloudy.runs))
    if colors is None:
        colors = ['red', 'blue', 'green']

    fig, axs = plt.subplots(1, 3, figsize=(21, 7))

    for r in run_indices:
        c = r % len(colors)

        c_run = cloudy.runs[r]
        s_run = skirt.runs[r]

        if c_run.name != s_run.name:
            raise ValueError(f"Run names do not match: {c_run.name} vs {s_run.name}")

        # Temperature
        c_R, c_T = c_run.temperature_profile()
        s_R, s_T = s_run.temperature_profile()
        if c_R.size and s_R.size:
            axs[0].plot(s_R, s_T, label=f"ski {c_run.name}", color=colors[c])
            axs[0].plot(c_R, c_T, label=f"cloudy {c_run.name}", linestyle='--', color=colors[c])
        axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        axs[0].set_xlabel("Radius (cm)")
        axs[0].set_ylabel("Temperature (K)")
        axs[0].legend()

        # SED
        c_wav, c_idx = c_run.load_cont_wav()
        c_inc, c_tra, c_emi, c_tot = c_run.load_cont(c_idx).T
        s_sed = s_run.load_sed()
        if s_sed.size:
            s_wav, s_tot, s_tra, s_dir = s_sed.T
            axs[1].plot(s_wav, s_dir, label=f"ski {c_run.name} (dir)", color=colors[c])
        axs[1].plot(c_wav, c_tra, label=f"cloudy {c_run.name} (dir)", linestyle='--', color=colors[c])
        axs[1].set_xscale('log')
        axs[1].set_yscale('log')
        axs[1].set_xlabel("Wavelength (m)")
        axs[1].set_ylabel("Flux (W/m2/m)")
        axs[1].legend()

        # Optical depth
        c_wl, _, c_dep, _ = c_run.load_depth()
        s_wl, s_dep = s_run.load_depth()
        if s_wl.size:
            axs[2].plot(s_wl, s_dep, label=f"ski {c_run.name}", color=colors[c])
        axs[2].plot(c_wl, c_dep, label=f"cloudy {c_run.name}", linestyle='--', color=colors[c])
        axs[2].set_xscale('log')
        axs[2].set_yscale('log')
        axs[2].set_xlabel("Wavelength (m)")
        axs[2].set_ylabel("Optical depth (1)")
        axs[2].legend()

    plt.show()


def plot_with_rad(fig, cloudy, skirt, bins, params, xquant, idy1=0, idy2=0, **kwargs):
    # --- prepare data and axes ---
    # yquant1 = stab1['quantityNames'][idy1]
    # yquant2 = stab2['quantityNames'][idy2]
    # x1 = stab1[xquant].value
    # x2 = stab2[xquant].value
    # y1 = stab1[yquant1].value
    # y2 = stab2[yquant2].value
    # min_y1 = np.nanmin(y1[y1 > 0])
    # max_y1 = np.nanmax(y1[y1 > 0])
    # min_y2 = np.nanmin(y2[y2 > 0])
    # max_y2 = np.nanmax(y2[y2 > 0])
    # axis_names = list(stab1['axisNames'])
    # if axis_names != list(stab2['axisNames']):
    #     raise ValueError("stab1 and stab2 must have the same axisNames")

    # create axes (kept positions from your original layout)
    ax1 = fig.add_axes([0.10, 0.10, 0.35, 0.60])
    ax2 = fig.add_axes([0.55, 0.10, 0.35, 0.60])
    ax3 = fig.add_axes([0.10, 0.77, 0.60, 0.18])

    # --- bins handling ---
    edges = np.asarray(bins.edges)
    num_bins = bins.num_bins

    keys = list(params.keys())
    bin_keys = []
    other_keys = []
    for key in keys:
        if key.startswith('bin'):
            bin_keys.append(key)
        else:
            other_keys.append(key)

    bin_vals = []
    other_vals = []
    for key in keys:
        if key.startswith('bin'):
            bin_vals.append(params[key])
        else:
            other_vals.append(params[key])

    if any(k != f"bin{idx}" for idx, k in enumerate(bin_keys)):
        raise ValueError(f"Expected bin keys named bin0..binN; got {bin_keys}")
    if edges.size != len(bin_keys) + 1:
        raise ValueError(f"edges length must be num_bins+1 ({len(bin_keys)+1}), got {edges.size}")

    sliders = {}

    slider_top = 0.9
    slider_height = 0.1

    i = 0
    for key, val in zip(other_keys, other_vals):
        # separate axis for ticks, placed slightly below
        nvals = len(val)
        ticks = [f"{v:.1e}" for v in val]
        ax_ticks = fig.add_axes([0.75, slider_top - i*slider_height - 0.001, 0.20, 0.02])
        ax_ticks.set_xlim(0, nvals-1)
        ax_ticks.set_xticks(range(nvals))
        ax_ticks.set_xticklabels(ticks, rotation=45, fontsize=7)
        ax_ticks.set_yticks([])
        for spine in ["top", "left", "right"]:
            ax_ticks.spines[spine].set_visible(False)

        # main slider axis
        ax_slider = fig.add_axes([0.75, slider_top - i*slider_height, 0.20, 0.02])
        sliders[key] = Slider(ax_slider, key, 0, nvals-1, valinit=0, valstep=1)

        i += 1

    # possible values for each bin
    bin_vals = [np.asarray(params[k]) for k in bin_keys]

    # initial value for each bin = first value
    current_vals = [bin_val[0] for bin_val in bin_vals]

    # centers for marker placement (geometric mean for log spacing)
    centers = np.sqrt(edges[:-1] * edges[1:])

    # --- helper to find index for a chosen bin value inside params[binKey] ---
    def find_value_index(bin_vals, current_vals):
        return int(np.argmin(np.abs(np.log10(bin_vals) - np.log10(current_vals))))

    # --- initial slicer and draw ax1/ax2 ---
    def make_indices():
        indices = []

        for key in bin_keys:
            idx = find_value_index(params[key], current_vals[bin_keys.index(key)])
            indices.append(idx)

        for key in other_keys:
            s = int(sliders[key].val)
            indices.append(s)

        return tuple(indices)

    indices = make_indices()
    idx = cloudy.get_index(indices)
    crun = cloudy.runs[idx]
    srun = skirt.runs[idx]

    # left plot
    c_R, c_T = crun.temperature_profile()
    s_R, s_T = srun.temperature_profile()
    s_R += 1e15  # add start point
    ax1_skirt, = ax1.plot(s_R, s_T, label=f"skirt", color='blue')
    ax1_cloudy, = ax1.plot(c_R, c_T, label=f"cloudy", linestyle='--', color='red')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel("Radius (cm)")
    ax1.set_ylabel("Temperature (K)")
    ax1.legend()

    # ax1_line, = ax1.plot(x1, yline1, **kwargs)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    # ax1.set_title(f"{yquant1}")
    # ax1.set_xlabel(f"{xquant} [{stab1['axisUnits'][stab1['axisNames'].index(xquant)]}]")
    # ax1.set_ylabel(f"{yquant1} [{stab1['quantityUnits'][stab1['quantityNames'].index(yquant1)]}]")
    # ax1.set_ylim(min_y1*0.8, max_y1*1.2)

    # right plot
    # ax2_line, = ax2.plot(x2, yline2, **kwargs)
    # ax2.set_xscale('log')
    # ax2.set_yscale('log')
    # ax2.set_title(f"{yquant2}")
    # ax2.set_xlabel(f"{xquant} [{stab2['axisUnits'][stab2['axisNames'].index(xquant)]}]")
    # ax2.set_ylabel(f"{yquant2} [{stab2['quantityUnits'][stab2['quantityNames'].index(yquant2)]}]")
    # ax2.set_ylim(min_y2*0.8, max_y2*1.2)

    for e in edges:
        # ax1.axvline(meV/e, color='gray', ls='--', lw=1)
        # ax2.axvline(meV/e, color='gray', ls='--', lw=1)
        ax3.axvline(e, color='gray', ls='--', lw=1)

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
    all_allowed = np.concatenate(bin_vals)
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

        indices = make_indices()
        idx = cloudy.get_index(indices)
        crun = cloudy.runs[idx]
        srun = skirt.runs[idx]

        c_R, c_T = crun.temperature_profile()
        s_R, s_T = srun.temperature_profile()

        ax1_skirt.set_ydata(s_T)
        ax1_cloudy.set_ydata(c_T)

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
        allowed = bin_vals[idx]
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
        on_motion(event)
        dragging['idx'] = None

    # connect events
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)

    def slider_update(val):
        update_main_plots()
    for s in sliders.values():
        s.on_changed(slider_update)

    # final initial draw
    update_main_plots()
    fig.canvas.draw_idle()
    return ax1, ax2, ax3, sliders
