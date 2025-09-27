import os
import re
import numpy as np
from astropy.io import fits
from ..cloudy.constants import meV


class Runs:
    def __init__(self, path=os.getcwd()):
        self.cwd = path
        self.runs_path = os.path.join(self.cwd, "ski")
        if not os.path.isdir(self.runs_path):
            raise FileNotFoundError(f"ski directory not found: {self.runs_path}")

    def discover(self):
        names = [d for d in os.listdir(self.runs_path) if os.path.isdir(
            os.path.join(self.runs_path, d)) and d.startswith("run")]
        names.sort()
        return [os.path.join(self.runs_path, n) for n in names]

    def load_runs(self, include_crashed=False):
        self.runs = []
        for p in self.discover():
            run = Run(p)
            if run.crashed() and not include_crashed:
                print(f"Skipping crashed SKI run: {run.path}")
                continue
            self.runs.append(run)
        print(f"Loaded {len(self.runs)} SKI runs from {self.runs_path}")


class Run:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        self.out = os.path.join(self.path, "out")

    def crashed(self):
        if not os.path.isdir(self.path):
            return True
        if not os.path.isdir(self.out):
            return True

        fn = os.path.join(self.out, "sim_log.txt")
        if not os.path.isfile(fn):
            return True

        with open(fn) as f:
            lines = f.readlines()
            if len(lines) < 2:
                return True
            if "Finished simulation" not in lines[-2]:
                return True
        return False

    def load_dat(self, filename, **kwargs):
        filename = os.path.join(self.path, filename)
        data = np.loadtxt(filename, comments="#", **kwargs)
        return np.atleast_1d(data)

    def load_temperature(self, **kwargs):
        return self.load_dat("out/sim_temp_gas_T.dat", **kwargs)

    def load_mix(self, **kwargs):
        return self.load_dat("mix.txt", **kwargs)

    # wav (m), tot, tra, dir (W/m2/m)
    def load_sed(self, **kwargs):
        sed = self.load_dat("out/sim_sed_sed.dat", usecols=(0, 1, 2, 3), **kwargs)
        sed[:, 0] = meV / sed[:, 0] * 1e-3  # eV
        sed[:, 1:] /= sed[:, 0][:, None]  # lam F_lam -> F_lam (W/m2/m)
        return sed

    def load_depth(self):
        # Load data
        with fits.open(os.path.join(self.out, "sim_opac_tau.fits")) as hdul:
            cube = np.array(hdul[0].data)  # shape (nx, ny, nw)
            wl = np.array(hdul[1].data['GRID_POINTS'])
            unit = hdul[1].columns['GRID_POINTS'].unit

        if unit == 'keV':
            wl = meV / wl * 1e-3
        elif unit == 'micron':
            wl *= 1-6
        return wl, cube[:, 0, 0]  # at pixel (0, 0)

    def temperature_profile(self):
        temp = self.load_temperature().T
        temp[0] *= 3.086e+18  # pc -> cm

        return temp[0], temp[1]
