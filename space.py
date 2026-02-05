from constants import *
import numpy as np


def ctran(points):
    """
    Transorm points to Cloudy space by applying log scaling to all columns except Z
    """
    mask = np.ones(points.shape[1], dtype=bool)
    mask[1] = False
    tran = points.copy()
    tran[:, mask] = np.log10(points[:, mask])
    tran[:, 0] *= HDEN_FACTOR
    tran[:, 1] *= METALLICITY_FACTOR
    tran[:, 2:] *= RAD_FACTOR
    return tran


def cdist(p1, p2):
    return np.abs(ctran(p1) - ctran(p2))


def dist(p1, p2):
    return np.abs(p1 - p2)
