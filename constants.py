"""
Here are some absorption edges I observed in the Cloudy opacity:
1.33593e-10 =	9280.740233395463 eV
2.53452e-09 =	489.18214494263214 eV
2.27717e-08 =	54.44661268153015 eV
9.11751e-08 =	13.598470744753776 eV

2.60459e-07 & 3.64756e-07 = 4.7602191899684785 eV & 3.399099480200463 eV
"""
numIons = 465
c = 2.99792458e8  # m/s
meV = 1.23984193e-6  # m <-> eV
Ryd = 13.6057039763  # eV
cts = 1e-7 * 1e4  # Cloudy -> SKIRT
nonZeroRange = (1e-1, 1e5)  # range used to limit the opac/emis
atomic_number = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
    'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
    'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
    'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30
}
