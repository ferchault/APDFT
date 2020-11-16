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
import scipy.optimize as sco
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
    # dm1_ao = calc.make_rdm1()
    # total_energy = calc.e_tot
    # Enn = calc.energy_nuc()

    # homocoeff = None
    # for idx, occ in enumerate(calc.mo_occ):
    #    if occ > 0:
    #        homocoeff = calc.mo_coeff[:, idx]

    # pyscf.tools.cubegen.orbital(calc.mol, f"idxhomod.cube", homocoeff)

    return calc


# calc = do_benzene(np.ones(6) + 6)
# calc2 = do_benzene(np.ones(6) + 6 + 0.01*np.array((1, -1, -1, 1, 0, 0)))
# calc3 = do_benzene(np.ones(6) + 6 + 0.001*np.array((1, -1, -1, 1, 0, 0)))


#%%
grid = pyscf.dft.gen_grid.Grids(calc.mol)
grid.level = 3
grid.build()
# ao = calc.mol.eval_gto("GTOval", np.array([[0., 0., 0.]]))
# %%

# %%
nmos = len(calc.mo_occ)
sim = np.zeros((nmos, nmos))
for i in range(nmos):
    psi_i = np.dot(ao, calc.mo_coeff[:, i])
    for j in range(nmos):
        psi_j = np.dot(ao, calc2.mo_coeff[:, j])
        prop = np.abs(psi_i) * np.abs(psi_j)
        sim[i, j] = np.sum(prop * grid.weights)
# %%
plt.imshow(sim)
# %%
sim[0]
# %%


def connect(origin, dest, calc_o=None, calc_d=None):
    print(origin[0], dest[0])
    if calc_o is None:
        calc_o = do_benzene(origin)
    if calc_d is None:
        calc_d = do_benzene(dest)

    nmos = len(calc.mo_occ)
    sim = np.zeros((nmos, nmos))
    for i in range(nmos):
        psi_i = np.abs(np.dot(ao, calc_o.mo_coeff[:, i]))
        for j in range(nmos):
            psi_j = np.abs(np.dot(ao, calc_d.mo_coeff[:, j]))
            prop = psi_i * psi_j
            sim[i, j] = np.sum(prop * grid.weights)

    row, col = sco.linear_sum_assignment(sim, maximize=True)
    scores = sim[row, col]

    if min(scores) < 0.9:
        print("insufficient similarity", min(scores))
        center = (origin + dest) / 2
        mapping_left = connect(origin, center, calc_o=calc_o)
        mapping_right = connect(center, dest, calc_d=calc_d)
        return mapping_left[mapping_right]
    else:
        return col


o = np.array((7, 5, 7, 5, 6, 6))
d = np.array((7, 5, 5, 7, 6, 6))
calc = do_benzene(o)
ao = pyscf.dft.numint.eval_ao(calc.mol, grid.coords, deriv=0)
grid = pyscf.dft.gen_grid.Grids(calc.mol)
grid.level = 3
grid.build()
connect(o, d)
#%%
len(set(np.argmax(sim, axis=1)))
# %%

# %%
row, col = sco.linear_sum_assignment(sim, maximize=True)

# %%
