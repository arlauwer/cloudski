import numpy as np
import os
import itertools
from .constants import *
from .bins import Bins
from pts.storedtable import writeStoredTable


def fake_table():

    edges = [1e5, 1e3, 1e1]  # eV
    bins = Bins(edges)
    num_bins = bins.num_bins

    axisGrids = {
        "n": np.array([1e15]),  # 1/m3
        "Z": np.array([0.5]),
        "bin0": np.array([1e4]),  # W/m2
        "bin1": np.array([1e5]),  # W/m2
    }
    param_shape = tuple(len(axisGrids[k]) for k in axisGrids)
    W = 3

    abundance = np.zeros((numIons, *param_shape))
    temperature = np.zeros(param_shape)
    opac = 1e-9 * np.ones((W, *param_shape))
    emis = 1e-9 * np.ones((W, *param_shape))

    opac[1] = 1e-8
    opac[2] = 1e-7

    ions = np.arange(numIons, dtype=int)
    wav = np.geomspace(meV/1e5, meV/1e-1, W)  # m
    bin_keys = [f'bin{i}' for i in range(num_bins)]

    print(wav)
    print(opac)

    writeStoredTable(
        "abund.stab",
        ["ion", "n", "Z"] + bin_keys,
        ["1", "1/m3", "1"] + ["W/m2"]*num_bins,
        ["lin", "lin", "lin"] + ["log"]*num_bins,
        [ions] + [axisGrids[k] for k in axisGrids.keys()],
        ["abund"],
        ["1/m3"],
        ["lin"],
        [abundance])

    writeStoredTable(
        "temp.stab",
        ["n", "Z"] + bin_keys,
        ["1/m3", "1"] + ["W/m2"]*num_bins,
        ["lin", "lin"] + ["log"]*num_bins,
        [axisGrids[k] for k in axisGrids.keys()],
        ["temp"],
        ["K"],
        ["lin"],
        [temperature])

    writeStoredTable(
        "opt.stab",
        ["lam", "n", "Z"] + bin_keys,
        ["m", "1/m3", "1"] + ["W/m2"]*num_bins,
        ["log", "lin", "lin"] + ["log"]*num_bins,
        [wav] + [axisGrids[k] for k in axisGrids.keys()],
        ["opac", "emis"],
        ["1/m", "W/m3"],
        ["lin", "lin"],
        [opac, emis])


# fake_table()
