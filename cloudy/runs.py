import os
import re
import json
import numpy as np
from .constants import *
from .bins import *
from .simgen import calc_sed
from pts.storedtable import writeStoredTable


class Runs:
    def __init__(self, path=os.getcwd()):
        self.cwd = path
        self.runs_path = os.path.join(self.cwd, "cloudy")
        self.params_path = os.path.join(self.runs_path, "params.json")

        self.load_params()
        self.create_grid()

    def load_params(self):
        if not os.path.exists(self.params_path):
            raise FileNotFoundError(f"Parameters file not found: {self.params_path}")

        with open(self.params_path) as f:
            self.json = json.load(f)
        self.params = self.json['params']
        self.num_params = len(self.params)
        self.num_runs = self.json['num_runs']
        self.bins = Bins(self.json['bins']['edges'])  # should be the exact same bins

    def create_grid(self):
        self.grid = np.load(os.path.join(self.runs_path, "grid.npy"))

    def get_index(self, indices):
        param_shape = self.grid.shape[1:]
        return np.ravel_multi_index(indices, param_shape)

    def get_param(self, run_idx):
        if run_idx < 0 or run_idx >= self.num_runs:
            raise IndexError("Run index out of range")

        idx = np.unravel_index(run_idx, self.grid.shape[1:])
        # prepend slice(None) for the first axis (keys)
        combo = self.grid[(slice(None),) + idx]

        return {key: val for key, val in zip(self.params.keys(), combo)}

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

    def export_skirt_sphe(self, outdir="ski"):
        with open("template/sphe.ski") as f:
            template = f.read()

        for r, run in enumerate(self.runs):
            param = self.get_param(r)

            R, depth, dr = run.load_zones()
            num_zones = len(R)
            leftR = R[0] + depth - dr
            rightR = R[0] + depth

            # mesh: normalize cumulative depths
            mesh = np.concatenate(([0], depth))
            mesh /= depth[-1]

            E, J_lambda, _, _ = calc_sed(self.bins, param)

            temp = template

            # luminosity
            F = sum(param[f'bin{b}'] for b in range(self.bins.num_bins))
            lum = F * 4 * np.pi * (param['rad'] * 1e-2)**2
            temp = temp.replace("{lum}", f"{lum} W")

            # radii
            temp = temp.replace("{minR}", f"{np.min(leftR)} cm")
            temp = temp.replace("{maxR}", f"{np.max(rightR)} cm")

            ski_path = os.path.join(outdir, run.name)
            os.makedirs(os.path.join(ski_path, "out"), exist_ok=True)

            # sim.ski
            with open(os.path.join(ski_path, "sim.ski"), "w") as f:
                f.write(temp)

            # sed.txt
            with open(os.path.join(ski_path, "sed.txt"), "w") as f:
                f.write("# Column 1: wavelength (eV)\n")
                f.write("# Column 2: specific luminosity (W/m)\n")
                for e, j in zip(E, J_lambda):
                    f.write(f"{e} {j}\n")

            # write mix.txt
            with open(os.path.join(ski_path, "mix.txt"), "w") as f:
                f.write("# Column 1: rmin (cm)\n")
                f.write("# Column 2: thetamin (deg)\n")
                f.write("# Column 3: phimin (deg)\n")
                f.write("# Column 4: rmax (cm)\n")
                f.write("# Column 5: thetamax (deg)\n")
                f.write("# Column 6: phimax (deg)\n")
                f.write("# Column 7: number density (1/cm3)\n")
                f.write("# Column 8: metallicity (1)\n")
                for i in range(num_zones):
                    hden = param['hden']
                    Z = param['Z']
                    f.write(f"{leftR[i]} {0} {0} {rightR[i]} {0} {0} {hden} {Z}\n")

            # write mesh.txt
            with open(os.path.join(ski_path, "mesh.txt"), "w") as f:
                for m in mesh:
                    f.write(f"{m}\n")

    def export_skirt_cart(self, outdir="ski", D=1e14):
        with open("template/cart.ski") as f:
            template = f.read()

        for r, run in enumerate(self.runs):
            param = self.get_param(r)

            R, depth, dr = run.load_zones()
            num_zones = len(R)
            minX = R[0] + depth - dr
            maxX = R[0] + depth
            minY = np.ones(num_zones) * -D
            maxY = np.ones(num_zones) * D
            minZ = np.ones(num_zones) * -D
            maxZ = np.ones(num_zones) * D

            # mesh: normalize cumulative depths
            mesh = np.concatenate(([0], depth))
            mesh /= depth[-1]

            E, J_lambda, _, J = calc_sed(self.bins, param)

            temp = template

            # luminosity
            lum = J * cts * (param['rad'] * 1e-2)**2
            temp = temp.replace("{lum}", f"{lum} W")

            # radii
            temp = temp.replace("{minX}", f"{np.min(minX)} cm")
            temp = temp.replace("{minY}", f"{np.max(minY)} cm")
            temp = temp.replace("{minZ}", f"{np.max(minZ)} cm")
            temp = temp.replace("{maxX}", f"{np.min(maxX)} cm")
            temp = temp.replace("{maxY}", f"{np.min(maxY)} cm")
            temp = temp.replace("{maxZ}", f"{np.min(maxZ)} cm")

            ski_path = os.path.join(outdir, run.name)
            os.makedirs(os.path.join(ski_path, "out"), exist_ok=True)

            # sim.ski
            with open(os.path.join(ski_path, "sim.ski"), "w") as f:
                f.write(temp)

            # sed.txt
            with open(os.path.join(ski_path, "sed.txt"), "w") as f:
                f.write("# Column 1: wavelength (eV)\n")
                f.write("# Column 2: specific luminosity (W/m)\n")
                for e, j in zip(E, J_lambda):
                    f.write(f"{e} {j}\n")

            # write mix.txt
            with open(os.path.join(ski_path, "mix.txt"), "w") as f:
                f.write("# Column 1: xmin (cm)\n")
                f.write("# Column 2: ymin (cm)\n")
                f.write("# Column 3: zmin (cm)\n")
                f.write("# Column 4: xmax (cm)\n")
                f.write("# Column 5: ymax (cm)\n")
                f.write("# Column 6: zmax (cm)\n")
                f.write("# Column 7: number density (1/cm3)\n")
                f.write("# Column 8: metallicity (1)\n")
                for i in range(num_zones):
                    hden = param['hden']
                    Z = param['Z']
                    f.write(f"{minX[i]} {minY[i]} {minZ[i]} {maxX[i]} {maxY[i]} {maxZ[i]} {hden} {Z}\n")

            # write mesh.txt
            with open(os.path.join(ski_path, "mesh.txt"), "w") as f:
                for m in mesh:
                    f.write(f"{m}\n")


#################### RUN ####################


class Run:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)

    def crashed(self):
        if not os.path.isdir(self.path):
            return True

        with open(os.path.join(self.path, "sim.out"), "rb") as f:
            f.seek(-18, 2)
            last_chars = f.read().decode()
            if last_chars != "Cloudy exited OK]\n":
                return True

        return False

    def load_cont_wav(self):
        E = np.loadtxt(os.path.join(self.path, "sim.con"), usecols=(0))
        idx = np.where((E >= nonZeroRange[0]) & (E <= nonZeroRange[1]))[0][::-1]
        return meV / E[idx], idx  # wavelength in m

    # incident, transmitted, emitted, total
    def load_cont(self, idx):
        # 4pi nuJnu in erg/s/cm2
        cont = np.loadtxt(os.path.join(self.path, "sim.con"), usecols=(0, 1, 2, 3, 6))
        cont = cont[idx]  # restrict to range
        m = meV / cont[:, 0]  # wavelength in m
        nuJnu = cont[:, 1:]  # erg/s/cm2
        Jlambda = nuJnu / (4 * np.pi * m[:, None])  # erg/s/cm2/m
        return Jlambda * cts  # erg/s/cm2/m -> W/m2/m

    def load_opac_wav(self):
        E = np.loadtxt(os.path.join(self.path, "sim.opac"), usecols=0)
        idx = np.where((E >= nonZeroRange[0]) & (E <= nonZeroRange[1]))[0][::-1]
        return meV / E[idx], idx  # wavelength in m

    def load_opac(self, idx):
        abs = np.loadtxt(os.path.join(self.path, "sim.opac"), usecols=2)
        return abs[idx] * 1e2  # 1/cm -> 1/m

    def load_depth(self):
        dep = np.loadtxt(os.path.join(self.path, "sim.depth"), usecols=(0, 1, 2, 3))
        E, tot, abs, sca = dep.T
        idx = np.where((E >= nonZeroRange[0]) & (E <= nonZeroRange[1]))[0]
        E = E[idx]
        tot = tot[idx]
        abs = abs[idx]
        sca = sca[idx]
        m = meV / E
        return m, tot, abs, sca

    def load_emis_wav(self):
        E = np.loadtxt(os.path.join(self.path, "sim.con"), usecols=(0))
        idx = np.where((E >= nonZeroRange[0]) & (E <= nonZeroRange[1]))[0][::-1]
        E = E[idx]
        return meV / E, idx  # wavelength in m

    def load_emis(self, idx):
        con = np.loadtxt(os.path.join(self.path, "sim.con"), usecols=(3, 8))
        diff_out, lin_out = con[:, 0], con[:, 1]
        emis = diff_out[idx] - lin_out[idx]  # 4pi nuJnu
        return emis * cts * 1e2 * 2  # erg/s/cm2 / 1cm -> W/m3
        # this only matches if also multiplied by 2 for some reason??? Otherwise it doesn't match the "save continuum emissivity"
        # perhaps spherical geometry in the save continuum emissivity? i.e. both sides of the cloud? -> so remove * 2

    def load_species(self):
        species_path = os.path.join(self.path, "sim.species")

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

        if data.ndim > 1:
            data = data.T

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
        overview = np.loadtxt(os.path.join(self.path, "sim.ovr"), skiprows=1, usecols=(1))
        overview = np.atleast_1d(overview)
        return overview

    def load_zones(self):
        # radius, depth (center of zone), dr
        zones = np.loadtxt(os.path.join(self.path, "sim.zones"), skiprows=1, usecols=(1, 2, 3))
        if zones.ndim == 1:
            zones = zones[np.newaxis, :]

        zones[:, 1] = np.cumsum(zones[:, 2])  # better definition of depth
        return zones.T

    def temperature_profile(self):
        temp = self.load_temperature()
        R, depth, dr = self.load_zones()
        Rmin = R[0] + depth - dr
        Rmax = R[0] + depth

        if temp.shape[0] != Rmin.shape[0]:
            raise ValueError(
                f"mismatch in number of zones: temperature has {temp.shape[0]} but zones has {Rmin.shape[0]}")

        n = Rmin.shape[0]
        R = np.zeros(2 * n)
        T = np.zeros(2 * n)
        for i in range(n):
            R[2*i], R[2*i+1] = Rmin[i], Rmax[i]
            T[2*i], T[2*i+1] = temp[i], temp[i]
        return R, T
