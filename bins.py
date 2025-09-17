from .constants import *
import numpy as np


class Bins:
    def __init__(self, edges):
        # assert edges are descending
        for i in range(len(edges)-1):
            assert edges[i] > edges[i+1]
            assert nonZeroRange[0] <= edges[i] <= nonZeroRange[1]

        self.edges = np.asarray(edges)
        self.num_bins = len(edges)-1
        self.widths = [meV/edges[i+1] - meV/edges[i] for i in range(len(edges)-1)]
