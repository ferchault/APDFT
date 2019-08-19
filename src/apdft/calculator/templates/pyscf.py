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
import basis_set_exchange as bse

angstrom = 1 / 0.52917721067

mol = pyscf.gto.Mole()
mol.atom = "{{ atoms }}"
mol.basis = {{basisset}}
mol.verbose = 0
mol.build()

method = "{{ method }}"
if method not in ["CCSD", "HF"]:
    raise NotImplementedError("Method %s not supported." % method)

deltaZ = np.array(({{deltaZ}}))
includeonly = np.array(({{includeonly}}))


def add_qmmm(calc, mol, deltaZ):
    mf = pyscf.qmmm.mm_charge(calc, mol.atom_coords() / angstrom, deltaZ)

    class NoSelfQMMM(mf.__class__):
        def energy_nuc(self):
            q = mol.atom_charges().astype(np.float)
            q[includeonly] += deltaZ
            return mol.energy_nuc(q)

    return NoSelfQMMM()


if method == "HF":
    calc = add_qmmm(pyscf.scf.RHF(mol), mol, deltaZ)
    hfe = calc.kernel(verbose=0)
    dm1_ao = calc.make_rdm1()
    total_energy = calc.e_tot
if method == "CCSD":
    calc = add_qmmm(pyscf.scf.RHF(mol), mol, deltaZ)
    hfe = calc.kernel(verbose=0)
    mycc = pyscf.cc.CCSD(calc).run()
    dm1 = mycc.make_rdm1()
    dm1_ao = np.einsum("pi,ij,qj->pq", calc.mo_coeff, dm1, calc.mo_coeff.conj())
    total_energy = mycc.e_tot

# GRIDLESS, as things should be ############################
# Total energy of SCF run

print("TOTAL_ENERGY", total_energy)

# Electronic EPN from electron density
for site in includeonly:
    mol.set_rinv_orig_(mol.atom_coords()[site])
    print("ELECTRONIC_EPN", site, np.matmul(dm1_ao, mol.intor("int1e_rinv")).trace())

# Electronic Dipole w.r.t to center of geometry
with mol.with_common_orig(mol.atom_coords().mean(axis=0)):
    ao_dip = mol.intor_symmetric("int1e_r", comp=3)
dipole = numpy.einsum("xij,ji->x", ao_dip, dm1_ao).real
print("ELECTRONIC_DIPOLE", *dipole)

# GRID, as things were #####################################
grid = pyscf.dft.gen_grid.Grids(mol)
grid.level = 3
grid.build()
ao_value = pyscf.dft.numint.eval_ao(mol, grid.coords, deriv=0)
rho = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_ao, xctype="LDA")

# Ionic Forces
for site in includeonly:
    rvec = grid.coords - mol.atom_coords()[site]
    force = [
        (rho * grid.weights * rvec[:, _] / np.linalg.norm(rvec, axis=1) ** 3).sum()
        for _ in range(3)
    ]
    print("IONIC_FORCE", site, *force)

# Quadrupole moments
rs = grid.coords - mol.atom_coords().mean(axis=0)
ds = np.linalg.norm(rs, axis=1) ** 2
# Q = np.zeros((3,3))
for i in range(3):
    for j in range(i, 3):
        q = 3 * rs[:, i] * rs[:, j]
        if i == j:
            q -= ds
        print("ELECTRONIC_QUADRUPOLE", i, j, (rho * q * grid.weights).sum())
