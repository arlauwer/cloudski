import os
import re
import json
import numpy as np
from .constants import *
from .bins import *
from pts.storedtable import writeStoredTable


class Runs:
    def __init__(self, path=os.getcwd()):
        self.cwd = path
        self.runs_path = os.path.join(self.cwd, "runs")
        self.params_path = os.path.join(self.runs_path, "params.json")

        self.load_params()
        self.create_grid()

    def load_params(self):
        if not os.path.exists(self.params_path):
            raise FileNotFoundError(f"Parameters file not found: {self.params_path}")

        with open(self.params_path) as f:
            self.json = json.load(f)
        self.params = self.json['params']
        self.num_runs = self.json['num_runs']
        self.bins = Bins(self.json['bins']['edges'])  # should be the exact same bins

    def create_grid(self):
        self.grid = np.load(os.path.join(self.runs_path, "grid.npy"))
        self.combos = self.grid.reshape(-1, self.grid.shape[-1])

    # DEPRECATED ### fix to use
    # def filter(self, filter):
    #     params = {}
    #     for dirname, param in self.json_runs.items():
    #         if any(param[key] not in mask for key, mask in filter.items()):
    #             continue

    #         params[dirname] = param

    #     self.json_runs = params
    #     self.create_grid()

    def load_runs(self, include_crashed=False):
        self.runs = []
        for r in range(self.num_runs):
            path = os.path.join(self.runs_path, f"run{r:05d}")
            run = Run(path)
            if run.crashed() and not include_crashed:
                print(f"Run {path} has crashed!")
                continue

            self.runs.append(run)

    def convert_to_stab(self, save_cont=False):
        num_bins = self.bins.num_bins

        ions = np.arange(numIons, dtype=int)
        wo, inRange_o = self.runs[0].load_opac_wav()
        we, inRange_e = self.runs[0].load_emis_wav()
        if save_cont:
            wc, inRange_c = self.runs[0].load_cont_wav()

        keys = list(self.params.keys())
        other_keys = keys[:-num_bins]
        bin_keys = keys[-num_bins:]

        vals = [np.array(v) for v in self.params.values()]
        other_vals = vals[:-num_bins]
        bin_vals = vals[-num_bins:]

        param_shape = self.grid.shape[1:]

        temperature = np.zeros(param_shape)
        abundance = np.zeros((numIons, *param_shape))
        opac = np.zeros((wo.size, *param_shape))
        emis = np.zeros((we.size, *param_shape))
        if save_cont:
            cont = np.zeros((wc.size, *param_shape, 4))  # incident, transmitted, emitted, total

        for run, idx in zip(self.runs, np.ndindex(param_shape)):
            temperature[idx] = run.load_temperature()
            abundance[:, *idx] = run.load_species()
            opac[:, *idx] = run.load_opac(inRange_o)
            emis[:, *idx] = run.load_emis(inRange_e)
            if save_cont:
                cont[:, *idx] = run.load_cont(inRange_c)

        stab_dir = os.path.join(self.cwd, "stab")
        os.makedirs(stab_dir, exist_ok=True)

        writeStoredTable(
            os.path.join(stab_dir, "temp.stab"),
            other_keys + bin_keys,
            ["1/m3", "1"] + ["W/m2"]*num_bins,
            ["lin", "lin"] + ["log"]*num_bins,
            other_vals + bin_vals,
            ["temp"],
            ["K"],
            ["lin"],
            [temperature])

        writeStoredTable(
            os.path.join(stab_dir, "abun.stab"),
            ["ion"] + other_keys + bin_keys,
            ["1", "1/m3", "1"] + ["W/m2"]*num_bins,
            ["lin", "lin", "lin"] + ["log"]*num_bins,
            [ions] + other_vals + bin_vals,
            ["abun"],
            ["1/m3"],
            ["lin"],
            [abundance]
        )

        writeStoredTable(
            os.path.join(stab_dir, "opac.stab"),
            ["lam"] + other_keys + bin_keys,
            ["m", "1/m3", "1"] + ["W/m2"]*num_bins,
            ["log", "lin", "lin"] + ["log"]*num_bins,
            [wo] + other_vals + bin_vals,
            ["opac"],
            ["1/m"],
            ["lin"],
            [opac]
        )

        writeStoredTable(
            os.path.join(stab_dir, "emis.stab"),
            ["lam"] + other_keys + bin_keys,
            ["m", "1/m3", "1"] + ["W/m2"]*num_bins,
            ["log", "lin", "lin"] + ["log"]*num_bins,
            [we] + other_vals + bin_vals,
            ["emis"],
            ["W/m3"],
            ["lin"],
            [emis]
        )

        if save_cont:
            writeStoredTable(
                os.path.join(stab_dir, "cont.stab"),
                ["lam"] + other_keys + bin_keys,
                ["m", "1/m3", "1"] + ["W/m2"]*num_bins,
                ["log", "lin", "lin"] + ["log"]*num_bins,
                [wc] + other_vals + bin_vals,
                ["inc", "tra", "emi", "tot"],
                ["W/m2", "W/m2", "W/m2", "W/m2"],
                ["lin", "lin", "lin", "lin"],
                [cont[..., 0], cont[..., 1], cont[..., 2], cont[..., 3]]
            )

#################### RUN ####################


class Run:
    def __init__(self, dirpath):
        self.dirpath = dirpath

    def crashed(self):
        if not os.path.isdir(self.dirpath):
            return True

        with open(os.path.join(self.dirpath, "sim.out"), "rb") as f:
            f.seek(-18, 2)
            last_chars = f.read().decode()
            if last_chars != "Cloudy exited OK]\n":
                return True

        return False

    def load_cont_wav(self):
        E = np.loadtxt(os.path.join(self.dirpath, "sim.con"), usecols=(0))
        idx = np.where((E >= nonZeroRange[0]) & (E <= nonZeroRange[1]))[0][::-1]
        return meV / E[idx], idx  # wavelength in m

    # incident, transmitted, emitted, total
    def load_cont(self, idx):
        # 4pi nuJnu in erg/s/cm2
        cont = np.loadtxt(os.path.join(self.dirpath, "sim.con"), usecols=(0, 1, 2, 3, 6))
        cont = cont[idx]  # restrict to range
        m = meV / cont[:, 0] # wavelength in m
        nuJnu = cont[:, 1:] # erg/s/cm2
        Jlambda = nuJnu / (4 * np.pi * m[:, None])  # erg/s/cm2/m
        return Jlambda * cts  # erg/s/cm2/m -> W/m2/m

    def load_opac_wav(self):
        E = np.loadtxt(os.path.join(self.dirpath, "sim.opac"), usecols=0)
        idx = np.where((E >= nonZeroRange[0]) & (E <= nonZeroRange[1]))[0][::-1]
        return meV / E[idx], idx  # wavelength in m

    def load_opac(self, idx):
        abs = np.loadtxt(os.path.join(self.dirpath, "sim.opac"), usecols=2)
        return abs[idx] * 1e4  # 1/cm -> 1/m

    def load_emis_wav(self):
        E = np.loadtxt(os.path.join(self.dirpath, "sim.con"), usecols=(0))
        idx = np.where((E >= nonZeroRange[0]) & (E <= nonZeroRange[1]))[0][::-1]
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
