import numpy as np
from constants import *
from read_probes import *


def read_opac(i):
    filename = OPAC_FILE(i)

    dat = np.loadtxt(filename)
    # We don't need to use the CELL_INDICES since the probe already accounts for this from its input
    # The first three columns are the positions, which can be discarded
    dat = dat[:, 3:]

    return dat


def read_opac_wav():
    """
    Read the wavelength grid of the opac from the header
    format: # column 4: opacity at E = 0.0136057 keV (1/pc)
    """
    wav = []

    filename = OPAC_FILE(0)
    with open(filename) as f:
        for line in f:
            if line.startswith("#") and line.endswith("keV (1/pc)\n"):
                wav.append(float(line.split()[7]))

    return np.array(wav)


def read_opacs():
    """
    (iterations, cells, opac)
    """
    num_iter = len(ALL_OPAC_FILES())

    opacs = []
    for i in range(num_iter):
        opacs.append(read_opac(i))

    return np.array(opacs)
