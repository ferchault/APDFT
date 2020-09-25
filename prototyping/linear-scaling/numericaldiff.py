#!/usr/bin/env python

# %%
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


def caseA():
    d = 2.7
    mol = pyscf.gto.Mole()
    mol.atom = f"H 0 0 0; H 0 0 {d};H 0 0 {2*d}; H 0 0 {3*d}; H 0 0 {4*d}; H 0 0 {5*d}"
    # mol.atom = "He 0 0 0; He 0 0 3"
    mol.basis = "def2-TZVP"
    mol.verbose = 0
    mol.build()
    calc = pyscf.scf.RHF(mol)
    print(calc.kernel() - calc.energy_nuc())
    dm = calc.make_rdm1()

    pos = np.zeros((1000, 3))
    pos[:, 2] = np.linspace(-2, 5 * d * angstrom + 2, 1000)
    ao_value = pyscf.dft.numint.eval_ao(mol, pos, deriv=0)
    rho = pyscf.dft.numint.eval_rho(mol, ao_value, dm, xctype="LDA")
    plt.semilogy(pos[:, 2], rho)


def add_qmmm(calc, mol, deltaZ):
    mf = pyscf.qmmm.mm_charge(calc, mol.atom_coords() / angstrom, deltaZ)

    def energy_nuc(self):
        q = mol.atom_charges().astype(np.float)
        q += deltaZ
        return mol.energy_nuc(q)

    mf.energy_nuc = energy_nuc.__get__(mf, mf.__class__)

    return mf


def caseB():
    d = 2.7
    mol = pyscf.gto.Mole()
    mol.atom = f"H 0 0 0; H 0 0 {d};H 0 0 {2*d}; H 0 0 {3*d}; H 0 0 {4*d}; H 0 0 {5*d}"
    # mol.atom = f"Li 0 0 {2*d}; Li 0 0 {3*d}"
    deltaZ = np.array([-1, -1, 2, 2, -1, -1])

    mol.basis = "def2-TZVP"
    mol.verbose = 0
    mol.build()

    calc = add_qmmm(pyscf.scf.RHF(mol), mol, deltaZ)
    print(calc.kernel() - calc.energy_nuc())
    dm = calc.make_rdm1()

    pos = np.zeros((1000, 3))
    pos[:, 2] = np.linspace(-2, 5 * d * angstrom + 2, 1000)
    ao_value = pyscf.dft.numint.eval_ao(mol, pos, deriv=0)
    rho = pyscf.dft.numint.eval_rho(mol, ao_value, dm, xctype="LDA")
    plt.semilogy(pos[:, 2], rho)


caseA()
caseB()

# %%
-2.0803224878294997e00 + -1.8862949396902806e-15 + -7.6217679233301017e-01 + -2.2573089231148202e-16 + -1.7390916703066150e-02

#%%
v = [
    float(_)
    for _ in """  -2.6147880983069376        
   1.8116471546186146         
   -0.03148695712063833       
   -0.04011522215717693       
   -0.05020162047412465       
   -0.06152725417280792       
   -0.0735767080589502        
   -0.08542318856844548       
   -0.09560682513016244       
   -0.10202550720912348       
    -0.10187013886274099      
    -0.09165151829923254      
    -0.06738283106417375      
    -0.02499708706118542      
    0.03891227316545354       
    0.12594456736433998       
    0.23428563261648927       
    0.3566565194811324        
    0.4779513438230736        
    0.5728838004840942""".split()
]
# %%
plt.semilogy(np.abs(np.cumsum(v[:-3]) - -1.2186083333428201))

# %%
plt.plot(v)
# %%
0.9 * angstrom
# %%
BNNB = np.array((-1, 1, 1, -1, 0, 0))
NBBN = np.array((1, -1, -1, 1, 0, 0))
BNBN = np.array((-1, 1, -1, 1, 0, 0))
np.outer(BNNB, BNNB) == np.outer(NBBN, NBBN)
# %%
np.average(
    np.abs(
        np.array((9.22 - 13.66, 6.4 - 2.24, 14.7 - 12.13, 20.5 - 16.57, 56.7 - 52.12))
    )
)
# %%
