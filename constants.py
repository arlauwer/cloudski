NUMBINS = 24
DIM = 2 + NUMBINS
LIBLOCATION = "/home/arno/Documents/UGent/PhD/run/ionised/knn/AGN/lib/"
OUTLOCATION = "/home/arno/Documents/UGent/PhD/run/ionised/knn/AGN/out/"

# Distance Factors
MAX_DIST = 0.5
HDEN_FACTOR = 1e0
METALLICITY_FACTOR = 1e0
RAD_FACTOR = 1e0

# Filenames
SKI = "AGN"
# Probenames
# PROBE_INFO = "conv" # unused
# PROBE_CUTS = "cuts" # unused
# PROBE_GRID = "grid" # unused
PROBE_OPAC = "opac"
PROBE_PROP = "prop"
PROBE_RWAV = "rad-wav"
PROBE_RGRID = "rad-grid"

# Cells
CELL_INDICES = slice(1, None, 4)  # splice of 1::4 to get a radially outwards line of cells with non-zero density
