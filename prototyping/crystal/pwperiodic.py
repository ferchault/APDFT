#!/usr/bin/env python
""" Reads debug VASP output and checks for EPN at nuclei."""

#%%
import matplotlib.pyplot as plt
import numpy as np

# %%
basename = "/mnt/c/Users/guido/Downloads/fort.129"
lines = open(basename).readlines()
linespart = open(basename + ".part").readlines()
# %%
ds = []
pots = []
for line in lines:
    parts = line.strip().split()
    if len(parts) != 3:
        continue
    if "." not in parts[0]:
        continue
    ds.append(float(parts[0]))
    pots.append(float(parts[1]))
# %%
plt.scatter(ds[:1875], pots[:1875])
plt.scatter(ds[1875:], pots[1875:], s=1)
# %%
len(ds) / 2
# %%
pot = []
potae = []
charges = []
weights = []
for line in linespart[::2]:
    parts = line.strip().split()
    pot.append(float(parts[1]))
    potae.append(float(parts[2]))
for line in linespart[1::2]:
    parts = line.strip().split()
    charges.append(float(parts[0]))
    weights.append(float(parts[1]))

# %%
plt.scatter(potae, pot)
# %%
plt.plot(sorted(ds), np.array(pots)[np.argsort(ds)])
# %%
(np.array(pot) * np.array(charges) * np.array(weights)).sum()
# %%
np.array(weights).sum()
# %%
pot[:10]
# %%
plt.plot(np.array(sorted(ds)), np.array(pots)[np.argsort(ds)])
plt.axvline(0)
# %%
np.array(pots)[np.argsort(ds)][:10]
# %%
sorted(ds)[:10]
# %%
plt.plot(charges)
# %%
plt.plot(np.array(weights) * np.array(charges))
# %%
import macrodensity as md
import numpy as np
import matplotlib.pyplot as plt

potential_file = (
    "/mnt/c/Users/guido/Documents/tmp/john-vasp/AECCAR2"
)  # The file with VASP output for potential

# Get the potential
vasp_pot, NGX, NGY, NGZ, Lattice = md.read_vasp_density(potential_file)
dx = Lattice[0, 0] / NGX

grid_pot, electrons = md.density_2_grid(vasp_pot, NGX, NGY, NGZ)
new_grid = np.array(grid_pot)
new_grid = new_grid[:, 54, :]
plt.plot(
    (np.arange(108) * dx - 4.5) * 1.8892, -new_grid[54], "o-"
)  # shift left site to origin, convert to bohr
plt.xlim(-2, 4)
# %%
electrons
# %%
from scipy.ndimage.filters import laplace

# %%
plt.plot(laplace(-new_grid[54]))
# %%
