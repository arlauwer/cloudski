import cloudy as cm
from cloudy.simgen import calc_sed
import numpy as np
import os

D = 1e14  # cm

cloudy = cm.Runs()
cloudy.load_runs(include_crashed=False)

# read template for sim.in
with open("template/sim.ski") as f:
    template = f.read()

for r, run in enumerate(cloudy.runs):
    # lum, hden, Z
    param = cloudy.get_param(r)

    # minX, maxX, minY, maxY, minZ, maxZ
    R, depth, dr = run.load_zones()
    num_zones = len(R)
    minX = R[0] + depth - dr
    maxX = R[0] + depth
    minY = np.ones(num_zones) * -D
    maxY = np.ones(num_zones) * D
    minZ = np.ones(num_zones) * -D
    maxZ = np.ones(num_zones) * D

    # mesh (all edges from minX to maxX)
    mesh = np.concatenate(([0], depth))
    mesh /= np.max(depth[-1])

    # sed
    E, J_lambda, J_nu, J = calc_sed(cloudy.bins, param)

    temp = template
    # lum
    temp = temp.replace("{lum}", str(param['lum']) + " erg/s")

    # minX, maxX, numX, minY, maxY, minZ, maxZ
    temp = temp.replace("{minX}", str(np.min(minX)) + " cm")
    temp = temp.replace("{maxX}", str(np.max(maxX)) + " cm")
    temp = temp.replace("{minY}", str(np.min(minY)) + " cm")
    temp = temp.replace("{maxY}", str(np.max(maxY)) + " cm")
    temp = temp.replace("{minZ}", str(np.min(minZ)) + " cm")
    temp = temp.replace("{maxZ}", str(np.max(maxZ)) + " cm")

    ski_path = os.path.join("ski", run.name)
    os.makedirs(ski_path, exist_ok=True)
    os.makedirs(os.path.join(ski_path, 'out'), exist_ok=True)

    # write sim.ski
    with open(os.path.join(ski_path, "sim.ski"), "w") as f:
        f.write(temp)

    # write sed.txt
    with open(os.path.join(ski_path, "sed.txt"), "w") as f:
        f.write("# Column 1: wavelength (eV)\n")
        f.write("# Column 2: specific luminosity (W/m2)\n")  # W/m2 or W is different??!
        for e, j in zip(E, J_lambda):
            f.write(f"{e} {j}\n")

    # write mix.txt
    with open(os.path.join(ski_path, "mix.txt"), "w") as f:
        f.write("# Column 1: xmin (cm)\n")
        f.write("# Column 2: ymin (cm)\n")
        f.write("# Column 3: zmin (cm)\n")
        f.write("# Column 4: xmax (cm)\n")
        f.write("# Column 5: ymax (cm)\n")
        f.write("# Column 6: zmax (cm)\n")
        f.write("# Column 7: number density (1/cm3)\n")
        f.write("# Column 8: metallicity (1)\n")
        for i in range(num_zones):
            hden = param['hden']
            Z = param['Z']
            f.write(f"{minX[i]} {minY[i]} {minZ[i]} {maxX[i]} {maxY[i]} {maxZ[i]} {hden} {Z}\n")

    # write mesh.txt
    with open(os.path.join(ski_path, "mesh.txt"), "w") as f:
        for m in mesh:
            f.write(f"{m}\n")
