import numpy as np
from constants import *
from read_probes import *
import matplotlib.pyplot as plt

""" AGN_prop_cellprops.dat """
# column 1: spatial cell index (1)
# column 2: x coordinate of cell center (pc)
# column 3: y coordinate of cell center (pc)
# column 4: z coordinate of cell center (pc)
# column 5: cell volume (pc3)
# column 6: optical depth of cell diagonal at 0.0137417 keV (1)
# column 7: dust mass density in cell (Msun/pc3)
# column 8: electron number density in cell (1/cm3)
# column 9: hydrogen number density in cell (1/cm3)


def all_props():
    return np.loadtxt(PROP_FILE(), usecols=(0, 1, 2, 3, 4, 5, 8))  # remove always zero columns


def radial_props():
    """
    Returns a list of cell properties of a radial profile (R = Rmin to Rmax)
    Without the duplicates along the theta direction.
    Originally built for a torus with Rbins=100 Thetabins=3(+1) -> this will return Rbins cells
    """
    allc = all_props()
    # There is 4 inclination bins per radius and the 2nd one (index 1) is the first one with a non-zero density
    return allc[1::4]


def all_cell_positions():
    allc = all_props()
    return allc[:, 1:4]


def radial_cell_positions():
    radc = radial_props()
    return radc[:, 1:4]


if __name__ == "__main__":
    line = radial_cell_positions()
    header = "Column 1: position x (pc)\nColumn 2: position y (pc)\nColumn 3: position z (pc)"
    np.savetxt("out/probe_positions.txt", line, header=header)

    x, y, z = all_cell_positions().T
    plt.scatter(x, z, label='all')
    plt.scatter(line[:, 0], line[:, 2], label='line')
    plt.legend()
    plt.show()
