import matplotlib as mpl
import os
from matplotlib.widgets import Slider
from joblib import Parallel, delayed
import json
import numpy as np
import itertools
import shutil
import glob
import re
import matplotlib.pyplot as plt
from pts.storedtable import writeStoredTable, readStoredTable, listStoredTableInfo
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


"""
Here are some absorption edges I observed in the Cloudy opacity:
1.33593e-10 =	9280.740233395463 eV
2.53452e-09 =	489.18214494263214 eV
2.27717e-08 =	54.44661268153015 eV
9.11751e-08 =	13.598470744753776 eV

2.60459e-07 & 3.64756e-07 = 4.7602191899684785 eV & 3.399099480200463 eV
"""

#################### CONSTANTS ####################

numIons = 465
c = 2.99792458e8  # speed of light in m/s
meV = 1.23984193e-6  # convert m <-> eV
Ryd = 13.6057039763  # Rydberg in eV
cts = 1e-7 * 1e4  # (Cloudy to SKIRT) convert erg/s/cm2 to W/m2
NonZeroRange = (1e-1, 1e5)  # energy range in eV for the opacity/emissivity tables

atomic_number = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30
}

#################### VARIABLES ####################

# edges = [1e5, 9.2801e3, 4.8918e2, 5.4466e1, Ryd, 1e-1]  # bin edges in eV (descending)
edges = [1e5, Ryd, 1e-1]  # bin edges in eV (descending)
assert (edges[0] > edges[-1]), "Edges (eV) must be in descending order."
num_bins = len(edges) - 1  # number of bins
m_edges = [meV / edge for edge in edges]  # bin edges in m
m_widths = [m_edges[i+1] - m_edges[i] for i in range(num_bins)]  # bin widths in m

#################### GENERATE ####################


def generate_bins(path, params):

    # Generate a binned SED based on the provided parameters.
    E = [edges[0]]
    J_lambda = [params['bin0'] / m_widths[0]]
    J = params['bin0']  # integrated W/m2

    for i in range(1, num_bins):
        # flat J_lambda value so that ∫ J_lambda dλ = param
        J += params[f'bin{i}']  # integrated W/m2
        J_flat = params[f'bin{i}'] / m_widths[i]

        E += [edges[i] * 1.001, edges[i] * 0.999]
        J_lambda += [J_lambda[-1], J_flat]

    E += [edges[-1]]
    J_lambda += [J_lambda[-1]]

    E = np.array(E, dtype=float)
    J_lambda = np.array(J_lambda, dtype=float)

    # Convert to wavelength [m] (ascending)
    m = meV / E

    # Convert for Cloudy output: J_nu [erg/s/cm2/Hz]
    J_nu = J_lambda / cts * m**2 / c

    # Write SED (descending E)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "cf.sed"), "w") as f:
        f.write("# Binned SED\n")
        for i, (e, j) in enumerate(zip(E, J_nu)):
            if i == 0:
                f.write(f"{e:.6g} {j:.6g} units eV\n")
            else:
                f.write(f"{e:.6g} {j:.6g}\n")
        f.write("**********\n")

    # convert to Cloudy J [erg/s/cm2]
    J = 4*np.pi * J / cts

    return J


def make_sim_dirs(params):
    if os.path.isdir("runs"):
        confirm = input("Do you want to delete the 'runs' directory? (y/n): ").strip().lower()
        if confirm == 'y':
            shutil.rmtree("runs")
            print("Deleted existing 'runs' directory.")
        elif confirm == 'f':
            print("Force continuing!!")
        else:
            print("Keeping existing 'runs' directory. Continuing.")
            return

    os.makedirs("runs", exist_ok=True)

    # verify params: count binx where x is number and make sure is the same as num_bins
    bin_keys = []
    for key, value in params.items():
        match = re.match(r'bin(\d+)', key)
        if match:
            bin_keys.append(key)
            bin_index = int(match.group(1))
            if not (0 <= bin_index < num_bins):
                raise ValueError(f"Invalid bin index for {key}: {value}")

    if len(bin_keys) != num_bins:
        raise ValueError(f"Number of bin parameters ({len(bin_keys)}) does not match expected num_bins ({num_bins}).")

    keys = params.keys()
    vals = params.values()
    combos = list(itertools.product(*vals))

    print(f"Creating {len(combos)} runs")

    all_params_dict = {}
    for idx, combo in enumerate(combos):
        combo_dict = dict(zip(keys, combo))

        run_name = f"run{idx:05d}"
        run_dir = os.path.join("runs", run_name)

        # SED
        os.makedirs(f"{run_dir}/SED", exist_ok=True)
        J = generate_bins(os.path.join(run_dir, "SED"), combo_dict)

        with open("template/sim.in") as f:
            template = f.read()
        sim_text = template
        # manually add intensity to be used in the Cloudy input files
        sim_text = sim_text.replace("{ins}", str(J))
        for k, v in combo_dict.items():
            sim_text = sim_text.replace(f"{{{k}}}", str(v))
        with open(os.path.join(run_dir, "sim.in"), "w") as f:
            f.write(sim_text)

        all_params_dict[run_name] = combo_dict

    # Save params.json
    with open(os.path.join("runs", "params.json"), "w") as f:
        json.dump(all_params_dict, f, indent=2)

    print("All simulation directories created successfully. Exiting.")
    exit(0)


#################### RUNS ####################


class Runs:
    def __init__(self):
        self.params_path = "runs/params.json"
        self.grid_path = "runs/grid.npz"

        self.load_params()
        self.create_grid()

    def load_params(self):
        """
        Loads all the parameters from the params.json file.
        This is generally only called once at the start.
        When using a filter the self.params dictionary is updated.
        This function will reset the self.params dictionary to the original state.
        """
        if not os.path.exists(self.params_path):
            raise FileNotFoundError(f"Parameters file not found: {self.params_path}")
        with open(self.params_path) as f:
            self.params = json.load(f)

    def create_grid(self):
        """
        Create a grid of all parameter combinations based on the self.params dictionary.
        The grid is a numpy array where each axis corresponds to a parameter.
        The first axis contains the actual data and has len(keys).
        eg. bin0 = [1, 2], bin1 = [10, 20]
        -> grid ~ 1+10, 1+20, 2+10, 2+20
        """
        # Consistent ordering of keys
        keys = list(next(iter(self.params.values())).keys())
        # Unique sorted values for each key
        vals = [sorted(set(p[k] for p in self.params.values())) for k in keys]

        # Shape: (len(keys), len(vals[0]), len(vals[1]), ...)
        shape = tuple(len(v) for v in vals) + (len(keys),)

        # All parameter combinations values (same way make_sim_dirs made the params.json!!!)
        self.combos = list(itertools.product(*vals))

        # Reshape combinations into a grid
        self.grid = np.array(self.combos, dtype=float).reshape(shape)
        self.grid = np.moveaxis(self.grid, -1, 0)  # Move len(keys) to the first axis

    def filter(self, filter):
        """
        Filter the self.params dictionary based on the provided filter.
        If a key is not present, it will be fully included.
        """
        params = {}
        for dirname, param in self.params.items():
            if any(param[key] not in mask for key, mask in filter.items()):
                continue

            params[dirname] = param

        self.params = params
        self.create_grid()

    def load_runs(self, include_crashed=False):
        """
        Load all the runs present in the self.params dictionary.
        Each run is represented by a Run object.
        """
        self.runs = []
        for dirname, param in self.params.items():
            full_path = os.path.join("runs", dirname)

            run = Run(full_path, param)
            if run.crashed() and not include_crashed:
                print(f"Run {dirname} has crashed!")
                continue

            self.runs.append(run)

    def convert_to_stab(self):
        ions = np.arange(numIons, dtype=int)
        wo, inRange_o = self.runs[0].load_opac_wav()
        we, inRange_e = self.runs[0].load_emis_wav()

        keys = list(next(iter(self.params.values())).keys())
        other_keys = keys[:-num_bins]
        bin_keys = keys[-num_bins:]

        others = {k: self.grid[i, *(slice(None) if j == i else 0 for j in range(len(keys)))]
                  for i, k in enumerate(other_keys)}
        bins = {k: self.grid[j, *(slice(None) if jj == j else 0 for jj in range(len(keys)))]
                for j, k in enumerate(bin_keys, start=len(other_keys))}

        param_shape = self.grid.shape[1:]

        temperature = np.zeros(param_shape)
        abundance = np.zeros((numIons, *param_shape))
        opac = np.zeros((wo.size, *param_shape))
        emis = np.zeros((we.size, *param_shape))

        for run, idx in zip(self.runs, np.ndindex(param_shape)):
            temperature[idx] = run.load_temperature()
            abundance[:, *idx] = run.load_species()
            opac[:, *idx] = run.load_opac(inRange_o)
            emis[:, *idx] = run.load_emis(inRange_e)

        writeStoredTable(
            "stab/temp.stab",
            ["n", "Z"] + bin_keys,
            ["1/m3", "1"] + ["W/m2"]*num_bins,
            ["lin", "lin"] + ["log"]*num_bins,
            list(others.values()) + list(bins.values()),
            ["temp"],
            ["K"],
            ["lin"],
            [temperature])

        writeStoredTable(
            "stab/abund.stab",
            ["ion", "n", "Z"] + bin_keys,
            ["1", "1/m3", "1"] + ["W/m2"]*num_bins,
            ["lin", "lin", "lin"] + ["log"]*num_bins,
            [ions] + list(others.values()) + list(bins.values()),
            ["abund"],
            ["1/m3"],
            ["lin"],
            [abundance]
        )

        writeStoredTable(
            "stab/opac.stab",
            ["lam", "n", "Z"] + bin_keys,
            ["m", "1/m3", "1"] + ["W/m2"]*num_bins,
            ["log", "lin", "lin"] + ["log"]*num_bins,
            [wo] + list(others.values()) + list(bins.values()),
            ["opac"],
            ["1/m"],
            ["lin"],
            [opac]
        )

        writeStoredTable(
            "stab/emis.stab",
            ["lam", "n", "Z"] + bin_keys,
            ["m", "1/m3", "1"] + ["W/m2"]*num_bins,
            ["log", "lin", "lin"] + ["log"]*num_bins,
            [we] + list(others.values()) + list(bins.values()),
            ["emis"],
            ["W/m3"],
            ["lin"],
            [emis]
        )

    def find_crashed_runs(self, delete=False):
        crashed = []

        for run in self.runs:
            if run.crashed():
                print(f"Run {run.dirpath} has crashed.")
                crashed.append(run.dirpath)

        # remove from params.json
        if delete:
            for dirname in crashed:
                self.params.pop(dirname, None)
            with open("runs/params.json", "w") as f:
                json.dump(self.params, f, indent=2)

#################### RUN ####################


class Run:
    def __init__(self, dirpath: str, params):
        self.dirpath = dirpath
        self.params = params

    def crashed(self):
        if not os.path.isdir(self.dirpath):
            return True

        with open(os.path.join(self.dirpath, "sim.out"), "rb") as f:
            f.seek(-18, 2)
            last_chars = f.read().decode()
            if last_chars != "Cloudy exited OK]\n":
                return True

        return False

    def load_con(self):
        """
        Load the sim.con file.
        Returns a numpy array with data = [E, incident, trans, diffout, net trans, reflected, total]
        """
        return np.loadtxt(os.path.join(self.dirpath, "sim.con"), skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6), dtype=float)

    def load_opac_wav(self):
        E = np.loadtxt(os.path.join(self.dirpath, "sim.opac"), usecols=0)
        idx = np.where((E >= NonZeroRange[0]) & (E <= NonZeroRange[1]))[0][::-1]
        return meV / E[idx], idx

    def load_opac(self, idx):
        abs = np.loadtxt(os.path.join(self.dirpath, "sim.opac"), usecols=2)
        return abs[idx] * 1e4  # 1/cm -> 1/m

    def load_emis_wav(self):
        E = np.loadtxt(os.path.join(self.dirpath, "sim.con"), usecols=(0))
        idx = np.where((E >= NonZeroRange[0]) & (E <= NonZeroRange[1]))[0][::-1]
        E = E[idx]
        return meV / E, idx  # wavelength in m

    def load_emis(self, idx):
        con = np.loadtxt(os.path.join(self.dirpath, "sim.con"), usecols=(3, 8))
        diff_out, lin_out = con[:, 0], con[:, 1]
        emis = diff_out[idx] - lin_out[idx]  # 4pi nuJnu
        return emis * cts * 1e2 * 2  # erg/s/cm2 / 1cm -> W/m3
        # this only matches if also multiplied by 2 for some reason??? Otherwise it doesn't match the "save continuum emissivity"
        # perhaps spherical geometry in the save continuum emissivity? i.e. both sides of the cloud? -> so remove * 2

    def load_species(self):
        species_path = os.path.join(self.dirpath, "sim.species")

        def parse_species(col):
            match = re.match(r'^([A-Z][a-z]?)(\+)?(\d+)?$', col)
            if match:
                element = match.group(1)
                charge = int(match.group(3)) if match.group(
                    2) and match.group(3) else (1 if match.group(2) else 0)
                return element, charge
            return None, None

        def is_valid_species(col):
            element, charge = parse_species(col)
            if element is None:
                return False
            return charge < atomic_number[element]

        with open(species_path, 'r') as file:
            header = np.array(file.readline().strip().split('\t'), dtype=str)

        # Filter valid species columns
        keep_cols = [i for i, col in enumerate(header) if is_valid_species(col)]
        labels = header[keep_cols]
        data = np.loadtxt(species_path, comments='#', usecols=keep_cols)

        # Prepare sorting keys: (Z, charge)
        sort_keys = []
        for col in labels:
            element, charge = parse_species(col)
            Z = atomic_number[element]
            sort_keys.append(Z*(Z-1) + charge)

        # Get sorted indices
        sorted_indices = np.argsort(sort_keys, axis=0)

        # Sort labels and data
        labels = labels[sorted_indices]
        data = data[sorted_indices]

        data *= 1e6  # 1/cm3 -> 1/m3

        return data[:numIons]

    def load_temperature(self):
        overview = np.loadtxt(os.path.join(self.dirpath, "sim.ovr"), skiprows=1)
        return overview[1]

    def load_sed(self):
        with open(os.path.join(self.dirpath, "SED/cf.sed"), "r") as f:
            lines = f.readlines()
            sed_data = []
            for line in lines:
                if line.strip() and not (line.startswith('#') or line.startswith('*')):
                    split = line.split()
                    sed_data.append([float(split[0]), float(split[1])])
            return np.array(sed_data)

    def plot_sed(self, ax):
        ax.set_xlabel("$E$ (eV)")
        ax.set_ylabel("$J_\\nu$ (erg/s/cm2/Hz)")
        sed = self.load_sed()
        E = sed[:, 0]
        J_nu = sed[:, 1]
        ax.plot(E, J_nu, label="SED", color='blue', marker='o')
        ax.legend()

    def plot_con(self, ax):
        ax.set_xlabel("$E$ (eV)")
        ax.set_ylabel("$J_\\lambda$ (erg/s/cm2/m)")
        ax.set_xscale("log")
        ax.set_yscale("log")

        con = self.load_con()
        E = con[:, 0]  # energy in eV

        t = self.load_emis2()

        # convert 4pi nu J_nu = 4pi lam J_lam -> J_lam (erg/s/cm2 -> W/m2/m)
        m = meV / E
        factor = cts / m / (4 * np.pi)
        factor = 1

        # FOR SOME REASON THE FACTOR IS STILL OFF BY 13.59 WHICH IS VERY CLOSE TO 1 RYDBERG???

        # ax.plot(E, con[:, 1] * factor, label="Incident", color='blue', linestyle='--', linewidth=0.5)
        # ax.plot(E, con[:, 2] * factor, label="Trans", color='orange', linestyle=':', linewidth=0.5)
        ax.plot(E, con[:, 3] * factor, label="Diffuse", color='green', linestyle='-', linewidth=0.2)
        ax.plot(E, t)
        # ax.plot(E, con[:, 4] * factor, label="Net Trans", color='red', linestyle=':', linewidth=0.5)
        # ax.plot(E, con[:, 5] * factor, label="Reflected", color='purple', linestyle=':', linewidth=0.5)
        # ax.plot(E, con[:, 6] * factor, label="Total", color='black', linestyle='-', linewidth=0.5)
        ax.legend()

    def plot_opt(self, ax):
        ax.set_xlabel("E (eV)")
        ax.set_ylabel("Emissivity (W/m3) and Opacity (1/m)")
        ax.set_xscale("log")
        ax.set_yscale("log")

        opt = self.load_emis()
        wav = opt[:, 0]
        E = meV / wav  # eV
        emis = opt[:, 1]  # erg/s/cm3
        kappa_abs = opt[:, 2]
        kappa_sca = opt[:, 3]

        j_lambda = emis / (4 * np.pi) / wav  # W/m3/m
        j_bol = np.trapz(j_lambda, wav)  # W/m3
        vol = 1.175e50  # m3
        L_bol = j_bol * vol
        print(f"L_bol = {L_bol:.3e} W = {L_bol/3.828e26:.3e} L_sun")

        ax.plot(E, emis, label="Emissivity", color='blue', linestyle='-')
        # ax.plot(E, kappa_abs, label="Absorption", color='orange', linestyle='--')
        # ax.plot(E, kappa_sca, label="Scattering", color='green', linestyle=':')
        ax.legend()

#################### MAIN ####################

params = {
    'hden': [1e8],
    'Z': [1.0],
    'bin0': [1e5],  # W/m2
    'bin1': [1e5],  # W/m2
    # 'bin2': [1e5, 1e9],  # W/m2
    # 'bin3': [1e5, 1e9],  # W/m2
    # 'bin4': [1e5, 1e9],  # W/m2
}
# make_sim_dirs(params)

### LOAD ###
cloudy = Runs()
cloudy.load_runs(include_crashed=False)

# fig, ax = plt.subplots(figsize=(10, 6))
# # cloudy.runs[0].plot_sed(ax)
# cloudy.runs[0].plot_con(ax)
# cloudy.runs[0].plot_opt(ax)
# plt.show()

cloudy.convert_to_stab()


### DANGER ###
# cloudy.find_crashed_runs(delete=True)
##############


### PLOTTING TABLE ###

def plot_single(ax, tab, xaxis, yquant, fixed=None, **kwargs):
    # ax.clear()
    x = tab[xaxis].value
    y = tab[yquant]

    fixed = fixed or {}

    slicer = []
    for a in tab['axisNames']:
        if a == xaxis:
            slicer.append(slice(None))
        elif a in fixed:
            slicer.append(fixed[a])
        else:
            slicer.append(0)
    yi = y[tuple(slicer)]

    ax.plot(x, yi.value, **kwargs)
    ax.set_xlabel(f"{xaxis} [{tab['axisUnits'][tab['axisNames'].index(xaxis)]}]")
    ax.set_ylabel(f"{yquant} [{tab['quantityUnits'][tab['quantityNames'].index(yquant)]}]")


# def plot_color_sweep(ax, tab, xaxis, yquant, coloraxis, fixed=None, cmap="rainbow_r", sm=None):
#     ax.clear()
#     x = tab[xaxis].value
#     y = tab[yquant]

#     cvals = tab[coloraxis].value
#     cmap_obj = plt.cm.get_cmap(cmap)
#     norm = mpl.colors.LogNorm(vmin=cvals.min(), vmax=cvals.max())

#     fixed = fixed or {}

#     for i, c in enumerate(cvals):
#         slicer = []
#         for a in tab['axisNames']:
#             if a == xaxis:
#                 slicer.append(slice(None))
#             elif a == coloraxis:
#                 slicer.append(i)
#             elif a in fixed:
#                 slicer.append(fixed[a])
#             else:
#                 slicer.append(0)
#         yi = y[tuple(slicer)]
#         ax.plot(x, yi.value, color=cmap_obj(norm(c)))

#     ax.set_xlabel(f"{xaxis} [{tab['axisUnits'][tab['axisNames'].index(xaxis)]}]")
#     ax.set_ylabel(f"{yquant} [{tab['quantityUnits'][tab['quantityNames'].index(yquant)]}]")

#     if sm is None:
#         sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
#         cbar = ax.figure.colorbar(sm, ax=ax)
#         cbar.set_label(f"{coloraxis} [{tab['axisUnits'][tab['axisNames'].index(coloraxis)]}]")
#         return sm
#     else:
#         sm.set_norm(norm)
#         return sm


# def interactive_bin_plot(tab, xaxis, yquant, sweep_bin, params):
#     bin_axes = [k for k in params if k.startswith("bin") and k != sweep_bin]
#     n_sliders = len(bin_axes)

#     # leave space for sliders
#     slider_height = 0.04
#     bottom_margin = 0.05 + n_sliders * slider_height
#     fig, ax = plt.subplots(figsize=(10, 6))
#     plt.subplots_adjust(left=0.25, bottom=bottom_margin)

#     fixed = {b: 0 for b in bin_axes}
#     sm = plot_color_sweep(ax, tab, xaxis, yquant, sweep_bin, fixed=fixed)

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
#         plot_color_sweep(ax, tab, xaxis, yquant, sweep_bin, fixed=fixed, sm=sm)
#         ax.set_xscale("log")
#         ax.set_yscale("log")
#         ax.set_ylim(1e-24, 1e-8)
#         fig.canvas.draw_idle()

#     for s in sliders.values():
#         s.on_changed(update)

#     plt.show()

listStoredTableInfo("stab_old/opt.stab")
opac = readStoredTable("stab/opac.stab")
emis = readStoredTable("stab/emis.stab")

print(opac["lam"].value[0])

fig, axs = plt.subplots(1, 2, figsize=(16, 8))
plot_single(axs[0], opac, "lam", "opac", fixed={"n": 0, "Z": 0, "bin0": 1, "bin1": 1}, label="opac", color='red')
plot_single(axs[0], emis, "lam", "emis", fixed={"n": 0, "Z": 0, "bin0": 1, "bin1": 1}, label="emis", color='blue')
axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].legend()
plt.show()

"""
# Compare old save emis with save sed
listStoredTableInfo("stab_old/opt.stab")
opt = readStoredTable("stab_old/opt.stab")
tab = readStoredTable("stab/emis.stab")

fig, axs = plt.subplots(1, 2, figsize=(16, 8))
plot_single(axs[0], opt, "lam", "emis", fixed={"n": 0, "Z": 0, "bin0": 1, "bin1": 1}, label="old", color='red')
plot_single(axs[0], tab, "lam", "emis", fixed={"n": 0, "Z": 0, "bin0": 1,
            "bin1": 1}, label="new", color='blue', linestyle='--')
axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].legend()
plt.show()
"""

# interactive_bin_plot(opt, "lam", "opac", "bin1", params)

# i = 6
# fig, axs = plt.subplots(1, 2, figsize=(24, 12))
# plot_color_sweep(axs[0], opt, "lam", "opac", "bin0", fixed={"n": 0, "Z": 0, "bin0": 0, "bin1": i})
# axs[0].set_ylim(1e-24, 1e-8)
# axs[0].set_xscale("log")
# axs[0].set_yscale("log")
# plt.title(f"Sweep bin0/bin1 index {i}")
# plot_color_sweep(axs[1], opt, "lam", "opac", "bin1", fixed={"n": 0, "Z": 0, "bin0": i, "bin1": 0})
# axs[1].set_xscale("log")
# axs[1].set_yscale("log")
# axs[1].set_ylim(1e-24, 1e-8)
# plt.show()


# frames = []
# for i in range(7):
#     fig, axs = plt.subplots(1, 2, figsize=(24, 12))
#     plot_color_sweep(axs[0], opt, "lam", "opac", "bin0", fixed={"n": 0, "Z": 0, "bin0": 0, "bin1": i})
#     axs[0].set_ylim(1e-24, 1e-8)
#     axs[0].set_xscale("log")
#     axs[0].set_yscale("log")
#     plt.title(f"Sweep bin0/bin1 index {i}")
#     plot_color_sweep(axs[1], opt, "lam", "opac", "bin1", fixed={"n": 0, "Z": 0, "bin0": i, "bin1": 0})
#     axs[1].set_xscale("log")
#     axs[1].set_yscale("log")
#     axs[1].set_ylim(1e-24, 1e-8)
#     fname = f"frame_{i}.png"
#     plt.title(f"Sweep bin0/bin1 index {i}")
#     fig.savefig(fname)
#     plt.close(fig)
#     frames.append(fname)

### FAKE TABLE ###

# def fake_table():

#     axisGrids = {
#         "n": np.array([1e15]),  # 1/m3
#         "Z": np.array([0.5]),
#         "bin0": np.array([1e4]),  # W/m2
#         "bin1": np.array([1e5]),  # W/m2
#     }
#     param_shape = tuple(len(axisGrids[k]) for k in axisGrids)
#     W = 3

#     abundance = np.zeros((numIons, *param_shape))
#     temperature = np.zeros(param_shape)
#     opac = 1e-9 * np.ones((W, *param_shape))
#     emis = 1e-9 * np.ones((W, *param_shape))

#     opac[1] = 1e-8
#     opac[2] = 1e-7

#     ions = np.arange(numIons, dtype=int)
#     wav = np.geomspace(meV/1e5, meV/1e-1, W)  # m
#     bin_keys = [f'bin{i}' for i in range(num_bins)]

#     print(wav)
#     print(opac)

#     writeStoredTable(
#         "abund.stab",
#         ["ion", "n", "Z"] + bin_keys,
#         ["1", "1/m3", "1"] + ["W/m2"]*num_bins,
#         ["lin", "lin", "lin"] + ["log"]*num_bins,
#         [ions] + [axisGrids[k] for k in axisGrids.keys()],
#         ["abund"],
#         ["1/m3"],
#         ["lin"],
#         [abundance])

#     writeStoredTable(
#         "temp.stab",
#         ["n", "Z"] + bin_keys,
#         ["1/m3", "1"] + ["W/m2"]*num_bins,
#         ["lin", "lin"] + ["log"]*num_bins,
#         [axisGrids[k] for k in axisGrids.keys()],
#         ["temp"],
#         ["K"],
#         ["lin"],
#         [temperature])

#     writeStoredTable(
#         "opt.stab",
#         ["lam", "n", "Z"] + bin_keys,
#         ["m", "1/m3", "1"] + ["W/m2"]*num_bins,
#         ["log", "lin", "lin"] + ["log"]*num_bins,
#         [wav] + [axisGrids[k] for k in axisGrids.keys()],
#         ["opac", "emis"],
#         ["1/m", "W/m3"],
#         ["lin", "lin"],
#         [opac, emis])


# fake_table()
