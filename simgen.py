import os
import itertools
import json
import numpy as np
from .constants import *
from .bins import Bins


def make_SED(path, bins, params):
    # bin edges and widths
    edges = bins.edges
    num_bins = bins.num_bins
    widths = bins.widths

    # leftmost edge
    E = [edges[0]]
    J_lambda = [params['bin0'] / widths[0]]
    J = params['bin0']

    # inner edges
    for i in range(1, num_bins):
        J += params[f'bin{i}']
        J_flat = params[f'bin{i}'] / widths[i]
        E += [edges[i] * 1.001, edges[i] * 0.999]
        J_lambda += [J_lambda[-1], J_flat]

    # rightmost edge
    E += [edges[-1]]
    J_lambda += [J_lambda[-1]]

    # convert to cloudy units
    E = np.array(E, dtype=float)
    J_lambda = np.array(J_lambda, dtype=float)
    m = meV / E
    J_nu = J_lambda / cts * m**2 / 3e8

    # write to file
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "cf.sed"), "w") as f:
        f.write("# Binned SED\n")
        for i, (e, j) in enumerate(zip(E, J_nu)):
            f.write(f"{e:.6g} {j:.6g}" + (" units eV\n" if i == 0 else "\n"))
        f.write("**********\n")

    # return integrated
    J = 4 * np.pi * J / cts
    return J


def make_sim_dirs(bins, params, path=os.getcwd()):

    # create runs directory
    run_path = os.path.join(path, "runs")
    os.makedirs(run_path, exist_ok=True)

    # list all parameter combinations
    keys = list(params.keys())
    vals = list(params.values())
    combos = np.array(list(itertools.product(*vals)))
    num_runs = combos.shape[0]

    # params.json structure
    json_dict = {
        "params": params,
        "num_runs": num_runs,
        "bins": {
            "edges": bins.edges,  # eV
            "widths": bins.widths,  # meter
            "num_bins": bins.num_bins
        }
    }

    # read template for sim.in
    template_path = os.path.join(path, "template/sim.in")
    with open(template_path) as f:
        template = f.read()

    for r, combo in enumerate(combos):
        # create run directory
        run_name = f"run{r:05d}"
        run_dir = os.path.join(run_path, run_name)
        sed_dir = os.path.join(run_dir, "SED")

        os.makedirs(sed_dir, exist_ok=True)

        # generate SED
        combo_dict = dict(zip(keys, combo))
        J = make_SED(sed_dir, bins, combo_dict)

        # write sim.in
        for k, v in combo_dict.items():
            template = template.replace(f"{{{k}}}", str(v))
        template = template.replace("{ins}", str(J))
        with open(os.path.join(run_dir, "sim.in"), "w") as f:
            f.write(template)

    # Shape: (len(keys), len(vals[0]), len(vals[1]), ...)
    shape = tuple(len(v) for v in vals) + (len(keys),)
    grid = np.array(combos, dtype=float).reshape(shape)
    grid = np.moveaxis(grid, -1, 0)  # Move len(keys) to the first axis
    np.save(os.path.join(run_path, "grid.npy"), grid)

    # save params.json
    with open(os.path.join(run_path, "params.json"), "w") as f:
        json.dump(json_dict, f, indent=2)
