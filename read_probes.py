from constants import *
import glob


def PROBE(probe):
    return OUTLOCATION+SKI+"_"+probe

# PROP


def PROP_FILE():
    return PROBE(PROBE_PROP)+"_cellprops.dat"


# RWAV

def RWAV_FILE():
    return PROBE(PROBE_RWAV)+"_wavelengths.dat"

# RGRID


def RGRID_FILE(i):
    return PROBE(PROBE_RGRID)+f"{i+1:03d}"+"_J.dat"


def ALL_RGRID_FILES():
    return glob.glob(PROBE(PROBE_RGRID)+"*_J.dat")
