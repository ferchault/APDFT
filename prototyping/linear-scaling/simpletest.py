#!/usr/bin/env python
# Linked to #228, the idea is to see whether close-lying FD DM versions are better approximations and converge with fixed cost

#%%
import numpy as np
import numpy
import pyscf
import pyscf.gto
import pyscf.qmmm
import pyscf.scf
import pyscf.dft
import pyscf.lib
from pyscf.data import nist
import matplotlib.pyplot as plt

# %%
angstrom = 1 / 0.52917721067
def add_qmmm(calc, mol, deltaZ):
    mf = pyscf.qmmm.mm_charge(calc, mol.atom_coords()/ angstrom, deltaZ)

    def energy_nuc(self):
        q = mol.atom_charges().astype(np.float)
        q += deltaZ
        return mol.energy_nuc(q)

    mf.energy_nuc = energy_nuc.__get__(mf, mf.__class__)

    return mf

class StepCounter:
    def update(self, locals):
        self.value = locals['cycle']

def get_dm(zs, iguess=None):
    counter = 0

    mol = pyscf.gto.Mole()
    mol.atom = "C -2.09726  2.41992 0;C -0.69947  2.47902 0;C  0.05061  1.29805 0;C -0.59710  0.05797 0;C -1.99490 -0.00113 0;C -2.74498  1.17984 0;H -0.19838  3.43838 0;H  1.13198  1.34377 0;H -0.01682 -0.85566 0;H -2.49598 -0.96049 0;H -3.82635  1.13412 0;H -2.67755  3.33356 0"
    mol.basis = "6-31G"
    mol.verbose = 0
    mol.build()

    deltaZ = np.concatenate((zs - mol.atom_charges().astype(np.float)[:6], np.zeros(6)))
    calc = add_qmmm(pyscf.scf.RHF(mol), mol, deltaZ)
    calc.conv_tol=1e-12
    counter = StepCounter()
    calc.callback = lambda _: StepCounter.update(counter, _)
    hfe = calc.kernel(verbose=0, dm0=iguess)
    dm1_ao = calc.make_rdm1()
    nsteps = counter.value
    return dm1_ao, nsteps
get_dm((6,6,6,6,6,6))

# %%
dmcenter, nsteps = get_dm((6.,6,6,6,6,6))
dm1up, nsteps = get_dm((6.05,6,6,6,6,6))
dm1dn, nsteps = get_dm((5.95,6,6,6,6,6))
dm2up, nsteps = get_dm((6, 6.05,6,6,6,6))
dm2dn, nsteps = get_dm((6, 5.95,6,6,6,6))
dm4up, nsteps = get_dm((6, 6, 6, 6.05,6,6))
dm4dn, nsteps = get_dm((6, 6,6,5.95,6,6))
dm2target, nsteps = get_dm((6.05,6.05,6,6,6,6))
print ("originally needed 2", nsteps)
dm4target, nsteps = get_dm((6.05,6,6,6.05,6,6))
print ("originally needed 4", nsteps)
# %%
get_dm((6.05,6.05,6,6.0,6,6), iguess=(dm1up-dmcenter + dm2up))[1]
get_dm((6.05,6.,6,6.05,6,6), iguess=(dm1up-dmcenter + dm4up))[1]
# %%
A, B, C, D, E = dmcenter, dm1up, dm1dn, dm4up, dm4dn
def score(weights):
    target = dm4target
    weights = weights / np.sum(weights)
    pred = sum([a*b for a, b in zip(weights, (A, B, C, D, E))])
    return get_dm((6.05,6.,6,6.05,6,6), iguess=pred)[1]
score((2,1,0,0,0))

# %%
import scipy.optimize as sco
weights = sco.minimize(score, x0=(2,1,0,0,0))
# %%
weights = weights / np.sum(weights)
est = sum([a*b for a, b in zip(weights, (A, B, C, D, E))])
# %%
est.shape
# %%
weights
# %%
dmcenter, nsteps = get_dm((6.,6,6,6,6,6))
dm1up, nsteps = get_dm((6.05,6,6,6,6,6))
print ("actual took", nsteps)
dm1up, nsteps = get_dm((6.05,6,6,6,6,6), iguess=2*dmcenter-1*dm1dn)
print ("estimate took", nsteps)

# %%
