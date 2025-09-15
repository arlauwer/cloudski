from .constants import *


class Bins:
    def __init__(self, edges):
        # assert edges are descending
        for i in range(len(edges)-1):
            assert edges[i] > edges[i+1]
            assert NonZeroRange[0] <= edges[i] <= NonZeroRange[1]

        self.edges = edges
        self.num_bins = len(edges)-1
        self.widths = [meV/edges[i+1] - meV/edges[i] for i in range(len(edges)-1)]
