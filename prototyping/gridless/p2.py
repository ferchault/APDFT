#!/usr/bin/env python
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.qmmm
import numpy as np

REFERENCE = (6, 8)
TARGET = (7, 7)
BASIS_SET = "def2-TZVP"

# get electronic energy difference and APDFT comparison
def add_qmmm(calc, mol, deltaZ):
    mf = pyscf.qmmm.mm_charge(calc, mol.atom_coords() * pyscf.data.nist.BOHR, deltaZ)

    def energy_nuc(self):
        q = mol.atom_charges().astype(np.float)
        q += deltaZ
        return mol.energy_nuc(q)

    mf.energy_nuc = energy_nuc.__get__(mf, mf.__class__)

    return mf


def build_mol(deltaZ):
    atomspec = "%d 0 0 0; %d 0 0 1" % (REFERENCE)
    mol = pyscf.gto.M(atom=atomspec, basis=BASIS_SET, verbose=0)
    calc = add_qmmm(pyscf.scf.RHF(mol), mol, deltaZ)
    return mol, calc


# reference molecule
refmol, refcalc = build_mol(np.zeros(2))

# target molecule in reference basis set
deltaZ = np.array(TARGET) - np.array(REFERENCE)
tarmol, tarcalc = build_mol(deltaZ)

# difference in nuclear-nuclear interaction
d = 1 / pyscf.data.nist.BOHR
delta_ENN = (np.prod(TARGET) - np.prod(REFERENCE)) / d

# electronic energy difference
actual_delta = tarcalc.kernel() - refcalc.kernel() - delta_ENN
print("Actual electronic energy difference:", -actual_delta)

# APDFT first orders for CO -> N2 (def2-TZVP)
orders = [-105.73009486303899, -108.87909018027341, -108.88796069507708]
print("ADFT1", orders[1] - orders[0])
print("ADFT2", orders[2] - orders[1])

# Density matrix expression
mol, calc = build_mol(np.zeros(2))
calc.kernel()
dm = calc.make_rdm1()

dV = 0
for site in (0, 1):
    with mol.with_rinv_origin(mol.atom_coords()[site]):
        dV += deltaZ[site] * mol.intor("int1e_rinv")

dm2 = np.einsum("ij,kl->ijkl", dm, dm) - np.einsum("ij,kl->ilkj", dm, dm)

# first order alchemical derivative (should be zero)
print("1st order", np.matmul(dm, dV).trace())

# second order
print("2nd order", np.matmul(0.5 * np.tensordot(dm2, dV), dV).trace())

