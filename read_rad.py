import numpy as np
import glob
from constants import *
from read_probes import *


def read_rad_wav():
    # left and right edges of the bins
    widths, lborder, rborder = np.loadtxt(RWAV_FILE(), usecols=(1, 2, 3)).T
    num_wav = lborder.size
    edges = np.zeros(num_wav+1)
    edges[0] = lborder[0]
    edges[1:] = rborder
    return edges, widths


def read_rad(i):
    filename = RGRID_FILE(i)

    dat = np.loadtxt(filename)

    return dat[CELL_INDICES, 1:]  # Remove cell indices


def read_rads():
    """
    (iterations, cells, bins)
    """
    num_iter = len(ALL_RGRID_FILES())

    rads = []
    for i in range(num_iter):
        rads.append(read_rad(i))

    return np.array(rads)
