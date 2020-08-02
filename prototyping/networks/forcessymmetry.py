#!/usr/bin/env python
# %%
import numpy as np
import apdft
import pyscf
import pyscf.gto
import pyscf.qmmm
import pyscf.scf
import pyscf.dft
import pyscf.lib
from pyscf.data import nist

# %%
angstrom = 1 / 0.52917721067


def changestr2delta(changestr):
    elements = "BCN"
    return np.array([elements.index(_) - 1 for _ in changestr])


def add_qmmm(calc, mol, deltaZ):
    mf = pyscf.qmmm.mm_charge(calc, mol.atom_coords() / angstrom, deltaZ)

    def energy_nuc(self):
        q = mol.atom_charges().astype(np.float)
        q += deltaZ
        return mol.energy_nuc(q)

    mf.energy_nuc = energy_nuc.__get__(mf, mf.__class__)

    return mf


def get_ionic_forces(charges, coordinates, changestr):
    deltaZ = changestr2delta(changestr)
    deltaZ = np.concatenate((deltaZ, np.zeros(len(charges) - len(deltaZ))))

    mol = pyscf.gto.Mole()
    atoms = []
    for c, coord in zip(charges, coordinates):
        atoms.append(" ".join([str(_) for _ in (c, *coord)]))
    mol.atom = ";".join(atoms)
    mol.basis = "def2-TZVP"
    mol.verbose = 0
    mol.build()

    # HF
    calc = add_qmmm(pyscf.scf.RHF(mol), mol, deltaZ)
    hfe = calc.kernel(verbose=0)
    dm1_ao = calc.make_rdm1()
    total_energy = calc.e_tot

    # grid
    grid = pyscf.dft.gen_grid.Grids(mol)
    grid.level = 3
    grid.build()
    ao_value = pyscf.dft.numint.eval_ao(mol, grid.coords, deriv=0)
    rho = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_ao, xctype="LDA")

    # ionic forces
    fs = []
    for site in range(len(charges)):
        rvec = grid.coords - mol.atom_coords()[site]
        force = [
            (rho * grid.weights * rvec[:, _] / np.linalg.norm(rvec, axis=1) ** 3).sum()
            for _ in range(3)
        ]
        fs.append((site, force))
    return fs


# %%
charges, coordinates = apdft.read_xyz("../../test/benzene.xyz")

# %%
center = get_ionic_forces(charges, coordinates, "CCCCCC")
up = get_ionic_forces(charges, coordinates, "BNBCNC")
dn = get_ionic_forces(charges, coordinates, "NBNCBC")

# %%
dn

# %%
import matplotlib.pyplot as plt

plt.rc("font", size=14)
fig = plt.figure(figsize=(4, 4))
colors = {"B": "red", "C": "darkgrey", "N": "blue", "H": "grey"}
plt.scatter(
    coordinates[:, 0],
    coordinates[:, 1],
    color=[colors[_] for _ in "BNBCNCHHHHHH"],
    s=50,
    zorder=10,
)
scaling = 3
for site in range(12):
    dx_up = (up[site][1][0] - center[site][1][0]) * scaling
    dy_up = (up[site][1][1] - center[site][1][1]) * scaling
    dx_dn = (dn[site][1][0] - center[site][1][0]) * scaling
    dy_dn = (dn[site][1][1] - center[site][1][1]) * scaling
    plt.plot(
        (coordinates[site, 0], coordinates[site, 0] + dx_up),
        (coordinates[site, 1], coordinates[site, 1] + dy_up),
        color="red",
    )
    plt.plot(
        (coordinates[site, 0], coordinates[site, 0] + dx_dn),
        (coordinates[site, 1], coordinates[site, 1] + dy_dn),
        color="blue",
    )
plt.xlabel("x [$\AA$]")
plt.ylabel("y [$\AA$]")
plt.savefig("ionic_forces.pdf", bbox_inches="tight")

# %%
