#!/usr/bin/env python
#%%
# region imports
import numpy as np
import numpy
import pyscf
import pyscf.gto
import pyscf.qmmm
import pyscf.scf
import pyscf.dft
import pyscf.lib
import pyscf.tools
from pyscf.data import nist
import matplotlib.pyplot as plt

# endregion
# %%
def do_benzene(zs):
    mol = pyscf.gto.Mole()
    mol.atom = """C  0.000000000000000  1.391100104090276  0.0
C  1.204728031075409  0.695550052045138 -0.0
C  1.204728031075409 -0.695550052045138 -0.0
C -0.000000000000000 -1.391100104090276  0.0
C -1.204728031075409 -0.695550052045138  0.0
C -1.204728031075409  0.695550052045138  0.0
H  0.000000000000000  2.471100189753489  0.0
H  2.140035536125550  1.235550092230858 -0.0
H  2.140035536125550 -1.235550092230858 -0.0
H -0.000000000000000 -2.471100189753489  0.0
H -2.140035536125550 -1.235550092230858  0.0
H -2.140035536125550  1.235550092230858  0.0"""
    mol.basis = "6-31G"
    mol.verbose = 0
    mol.build()

    deltaZ = np.array(zs) - 6
    includeonly = np.arange(6)

    def add_qmmm(calc, mol, deltaZ):
        mf = pyscf.qmmm.mm_charge(
            calc, mol.atom_coords()[includeonly] * nist.BOHR, deltaZ
        )

        def energy_nuc(self):
            q = mol.atom_charges().astype(np.float)
            q[includeonly] += deltaZ
            return mol.energy_nuc(q)

        mf.energy_nuc = energy_nuc.__get__(mf, mf.__class__)

        return mf

    calc = add_qmmm(pyscf.scf.RHF(mol), mol, deltaZ)
    hfe = calc.kernel(verbose=0)
    dm1_ao = calc.make_rdm1()
    total_energy = calc.e_tot
    Enn = calc.energy_nuc()

    homocoeff = None
    for idx, occ in enumerate(calc.mo_occ):
        if occ > 0:
            homocoeff = calc.mo_coeff[:, idx]

    pyscf.tools.cubegen.orbital(calc.mol, f"idxhomod.cube", homocoeff)

    return calc


calc = do_benzene(np.ones(6) + 6)


#%%
