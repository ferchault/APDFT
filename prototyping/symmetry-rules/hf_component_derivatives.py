#!/usr/bin/env python
# %%
# H2 test case as obtained from Gaussian
##HF/6-31G  ExtraLinks=L608
#
# title
#
# 0 1
# H 0. 0. 0.
# H 0. 0. 1.
# Kinetic          0.924150
# electron-nuclei -3.117033
# Coulomb          1.137794
# HFX             -0.568897
# Nuclear-nuclear  0.529177

import numpy as np
import numpy
import pyscf
import pyscf.gto
import pyscf.qmmm
import pyscf.scf
import pyscf.dft
import pyscf.lib
from pyscf.data import nist
import pandas as pd
import basis_set_exchange as bse

angstrom = 1 / 0.52917721067


def get_components(molstring, basisset, deltaZ, includeonly):
    mol = pyscf.gto.Mole()
    mol.atom = molstring
    mol.basis = basisset
    mol.verbose = 0
    mol.build()
    hen = mol.intor_symmetric("int1e_nuc")

    def add_qmmm(calc, mol, deltaZ):
        mf = pyscf.qmmm.mm_charge(
            calc, mol.atom_coords()[includeonly] / angstrom, deltaZ
        )

        def energy_nuc(self):
            q = mol.atom_charges().astype(np.float)
            q[includeonly] += deltaZ
            return mol.energy_nuc(q)

        mf.energy_nuc = energy_nuc.__get__(mf, mf.__class__)
        return mf

    calc = add_qmmm(pyscf.scf.RHF(mol), mol, deltaZ)
    hfe = calc.kernel(verbose=0)
    dm = calc.make_rdm1()
    total_energy = calc.e_tot

    hkin = mol.intor_symmetric("int1e_kin")
    # hen = mol.intor_symmetric("int1e_nuc")
    vhf = calc.get_veff(mol, dm)

    e_nn = calc.energy_nuc()
    kinetic = numpy.einsum("ij,ji->", hkin, dm)
    en = numpy.einsum("ij,ji->", hen, dm)
    hfx = numpy.einsum("ij,ji->", vhf, dm) * 0.5
    return {"total": total_energy, "nn": e_nn, "kin": kinetic, "en": en, "hfx": hfx}


direction = np.array((1, -1))
results = []
for lval in np.linspace(0, 1, 10):
    results.append(
        get_components("H 0 0 0; H 0 0 1", "6-31G", direction * lval, [0, 1])
    )
results = pd.DataFrame(results)

# %%
import matplotlib.pyplot as plt

for col in results.columns:
    plt.plot(results[col], label=col)
plt.legend()
# %%
plt.plot(results[:, 3])
plt.plot(results[:, 0] - results[:, 1] - results[:, 2] - results[:, 4])
# %%
get_components("He 0 0 0", bse.get_basis("6-31G", 1, fmt="nwchem"), np.array([0]), [0])
# %%
results
# %%
np.linspace(0, 1, 10)
# %%


# %%
