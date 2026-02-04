import struct
import numpy as np
from constants import *

def to_name(i):
	if i == 0:
		return 'n'
	elif i == 1:
		return 'z'
	else:
		return f'rad {i-2}'

def read_hnsw_points(filename=LIBLOCATION+"hnsw.bin", dim=DIM):
    with open(filename, "rb") as f:
        # --- header ---
        offsetLevel0      = struct.unpack("<Q", f.read(8))[0]
        max_elements      = struct.unpack("<Q", f.read(8))[0]
        cur_elements      = struct.unpack("<Q", f.read(8))[0]
        size_per_element  = struct.unpack("<Q", f.read(8))[0]
        label_offset      = struct.unpack("<Q", f.read(8))[0]
        offsetData        = struct.unpack("<Q", f.read(8))[0]
        maxlevel          = struct.unpack("<i", f.read(4))[0]
        enterpoint_node   = struct.unpack("<I", f.read(4))[0]
        maxM              = struct.unpack("<Q", f.read(8))[0]

        maxM0             = struct.unpack("<Q", f.read(8))[0]
        M                 = struct.unpack("<Q", f.read(8))[0]
        mult              = struct.unpack("<d", f.read(8))[0]
        ef_construction   = struct.unpack("<Q", f.read(8))[0]

        # --- read data block ---
        raw = f.read(cur_elements * size_per_element)

    vec_bytes = dim * 8
    points = np.zeros((cur_elements, dim), dtype=np.float64)

    for i in range(cur_elements):
        start = i * size_per_element + offsetData
        end   = start + vec_bytes
        points[i] = np.frombuffer(raw[start:end], dtype=np.float64)

    return points
