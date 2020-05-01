import numpy as np
import numpy
import pyscf
import pyscf.gto
import pyscf.qmmm
import pyscf.scf
import pyscf.dft
import pyscf.lib
from pyscf.data import nist

angstrom = 1 / 0.52917721067

mol = pyscf.gto.Mole()
mol.atom = "8 0.000000 0.000000 0.000000;1 0.000000 1.000000 0.000000;1 1.000000 0.000000 0.000000"
mol.basis = {8: '#----------------------------------------------------------------------\n# Basis Set Exchange\n# Version v0.8.12\n# https://www.basissetexchange.org\n#----------------------------------------------------------------------\n#   Basis set: def2-TZVP\n# Description: def2-TZVP\n#        Role: orbital\n#     Version: 1  (Data from Turbomole 7.3)\n#----------------------------------------------------------------------\n\n\nBASIS "ao basis" PRINT\n#BASIS SET: (11s,6p,2d,1f) -> [5s,3p,2d,1f]\nO    S\n  27032.3826310              0.21726302465E-03\n   4052.3871392              0.16838662199E-02\n    922.32722710             0.87395616265E-02\n    261.24070989             0.35239968808E-01\n     85.354641351            0.11153519115\n     31.035035245            0.25588953961\nO    S\n     12.260860728            0.39768730901\n      4.9987076005           0.24627849430\nO    S\n      1.1703108158           1.0000000\nO    S\n      0.46474740994          1.0000000\nO    S\n      0.18504536357          1.0000000\nO    P\n     63.274954801            0.60685103418E-02\n     14.627049379            0.41912575824E-01\n      4.4501223456           0.16153841088\n      1.5275799647           0.35706951311\nO    P\n      0.52935117943           .44794207502\nO    P\n      0.17478421270           .24446069663\nO    D\n      2.31400000             1.0000000\nO    D\n      0.64500000             1.0000000\nO    F\n      1.42800000             1.0000000\nEND\n', 1: '#----------------------------------------------------------------------\n# Basis Set Exchange\n# Version v0.8.12\n# https://www.basissetexchange.org\n#----------------------------------------------------------------------\n#   Basis set: def2-TZVP\n# Description: def2-TZVP\n#        Role: orbital\n#     Version: 1  (Data from Turbomole 7.3)\n#----------------------------------------------------------------------\n\n\nBASIS "ao basis" PRINT\n#BASIS SET: (5s,1p) -> [3s,1p]\nH    S\n     34.0613410              0.60251978E-02\n      5.1235746              0.45021094E-01\n      1.1646626              0.20189726\nH    S\n      0.32723041             1.0000000\nH    S\n      0.10307241             1.0000000\nH    P\n      0.8000000              1.0000000\nEND\n'}
mol.verbose = 0
mol.build()

method = "CCSD"
if method not in ["CCSD", "HF"]:
    raise NotImplementedError("Method %s not supported." % method)

deltaZ = np.array((0.0,0.0,0.0))
includeonly = np.array((0,1,2))


def add_qmmm(calc, mol, deltaZ):
    mf = pyscf.qmmm.mm_charge(calc, mol.atom_coords()[includeonly] / angstrom, deltaZ)

    def energy_nuc(self):
        q = mol.atom_charges().astype(np.float)
        q[includeonly] += deltaZ
        return mol.energy_nuc(q)

    mf.energy_nuc = energy_nuc.__get__(mf, mf.__class__)
    
    return mf


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