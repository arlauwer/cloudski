import os
import re
import json
import itertools
import numpy as np
from .constants import *
from pts.storedtable import writeStoredTable


class Runs:
    def __init__(self, path=os.getcwd()):
        self.runs_path = os.path.join(path, "runs")
        self.params_path = os.path.join(self.runs_path, "params.json")

        self.load_params()
        self.create_grid()

    def load_params(self):
        if not os.path.exists(self.params_path):
            raise FileNotFoundError(f"Parameters file not found: {self.params_path}")
        with open(self.params_path) as f:
            self.json = json.load(f)

        self.runs_dict = self.json['runs']
        self.run_vals = list(self.runs_dict.values())
        self.bins_dict = self.json['bins']

    def create_grid(self):
        # Consistent ordering of keys
        keys = list(next(iter(self.run_vals)).keys())
        # Unique sorted values for each key
        vals = [sorted(set(p[k] for p in self.run_vals)) for k in keys]

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
        for dirname, param in self.runs_dict.items():
            if any(param[key] not in mask for key, mask in filter.items()):
                continue

            params[dirname] = param

        self.runs_dict = params
        self.create_grid()

    def load_runs(self, include_crashed=False):
        """
        Load all the runs present in the self.params dictionary.
        Each run is represented by a Run object.
        """
        self.runs = []
        for dirname, param in self.runs_dict.items():
            full_path = os.path.join(self.runs_path, dirname)

            run = Run(full_path, param)
            if run.crashed() and not include_crashed:
                print(f"Run {dirname} has crashed!")
                continue

            self.runs.append(run)

    def convert_to_stab(self):
        num_bins = self.bins_dict['num_bins']

        ions = np.arange(numIons, dtype=int)
        wo, inRange_o = self.runs[0].load_opac_wav()
        we, inRange_e = self.runs[0].load_emis_wav()

        keys = list(next(iter(self.run_vals)).keys())
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
            "stab/abun.stab",
            ["ion", "n", "Z"] + bin_keys,
            ["1", "1/m3", "1"] + ["W/m2"]*num_bins,
            ["lin", "lin", "lin"] + ["log"]*num_bins,
            [ions] + list(others.values()) + list(bins.values()),
            ["abun"],
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

    def find_crashed_runs(self):
        crashed = []

        for run in self.runs:
            if run.crashed():
                print(f"Run {run.dirpath} has crashed.")
                crashed.append(run.dirpath)

#################### RUN ####################


class Run:
    def __init__(self, dirpath, params):
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

        # t = self.load_emis2()

        # convert 4pi nu J_nu = 4pi lam J_lam -> J_lam (erg/s/cm2 -> W/m2/m)
        m = meV / E
        factor = cts / m / (4 * np.pi)
        # factor = 1

        # FOR SOME REASON THE FACTOR IS STILL OFF BY 13.59 WHICH IS VERY CLOSE TO 1 RYDBERG???

        # ax.plot(E, con[:, 1] * factor, label="Incident", color='blue', linestyle='--', linewidth=0.5)
        # ax.plot(E, con[:, 2] * factor, label="Trans", color='orange', linestyle=':', linewidth=0.5)
        ax.plot(E, con[:, 3] * factor, label="Diffuse", color='green', linestyle='-', linewidth=0.2)
        # ax.plot(E, t)
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
