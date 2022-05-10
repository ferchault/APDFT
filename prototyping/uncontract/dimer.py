#%%
import pyscf.gto
import pyscf.scf
import pyscf.qmmm
import numpy as np

# %%

def add_qmmm(calc, mol, deltaZ):
    """ Alters the effective nuclear charges by adding point charges on top of atoms."""
    angstrom = 1 / 0.52917721067
    mf = pyscf.qmmm.mm_charge(calc, mol.atom_coords() / angstrom, deltaZ)

    def energy_nuc(self):
        q = mol.atom_charges().astype(float)
        q += np.array(deltaZ)
        return mol.energy_nuc(q)

    mf.energy_nuc = energy_nuc.__get__(mf, mf.__class__)

    return mf


def get_N2_energy(dZ, basis):
    """ Calculates the electronic N2 energy with changes in nuclear charges dZ in the given basis"""
    mol = pyscf.gto.M(atom="N 0 0 0; N 0 0 1", basis=basis)
    mol.verbose = 0
    mf = pyscf.scf.RHF(mol)
    calc = add_qmmm(mf, mol, dZ)
    return calc.kernel() - calc.energy_nuc()


def get_3rd_order_CO_estimate(basis):
    """ Finite differences for up to including 3rd order."""
    delta = 0.001
    dlambda = [_ * delta for _ in range(-2, 3)]
    energies = [get_N2_energy((_, -_), basis) for _ in dlambda]

    zeroth = energies[2]
    first = (0.5*energies[3] - 0.5*energies[1]) / delta
    second = (energies[1] + energies[3] - 2 * energies[2]) / delta**2
    third = (-0.5*energies[0] + energies[1] - energies[3]+0.5*energies[4]) / delta**3

    return zeroth+first+second/2+third/6

def get_true_CO_estimate(basis):
    """ CO electronic energy in the basis set of N2."""
    return get_N2_energy((1, -1), basis)

def get_error_reference_basis(basis):
    """ Compares prediction to SCF in N2 basis set."""
    return (get_3rd_order_CO_estimate(basis) - get_true_CO_estimate(basis))*1000

def get_error_target_basis(basisname, basis):
    """ Compares prediction to SCF in CO basis set."""
    mol = pyscf.gto.M(atom="C 0 0 0; O 0 0 1", basis=basisname)
    mol.verbose = 0
    calc = pyscf.scf.RHF(mol)
    expected = calc.kernel() - calc.energy_nuc()
    got = get_3rd_order_CO_estimate(basis)
    return (got - expected)*1000

for basis in "STO-3G 6-31G def2-TZVP".split():
    print ("#"*50)
    uncontracted = pyscf.gto.uncontract(pyscf.gto.load(basis, "N"))

    print ("compared to basis set of the reference")
    print ("  contracted", basis, get_error_reference_basis(basis), "mHa")
    print ("uncontracted", basis, get_error_reference_basis(uncontracted), "mHa")
    print ("")
    print ("compared to basis set of the target")
    print ("  contracted", basis, get_error_target_basis(basis, basis), "mHa")
    print ("uncontracted", basis, get_error_target_basis(basis, uncontracted), "mHa")

# %%

# %%
