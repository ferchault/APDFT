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
class ConnectedBenzene:
    def _get_mol(self):
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
        return mol

    def _do_run(self, lval):
        mol = self._get_mol()
        zs = lval * self._meta_direction + self._meta_origin

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

        self._calcs[lval] = calc
        return calc

    def _connect(self, origin, dest, calc_o=None, calc_d=None):
        if calc_o is None:
            calc_o = self._do_run(origin)
        if calc_d is None:
            calc_d = self._do_run(dest)

        nmos = len(calc_o.mo_occ)
        sim = np.zeros((nmos, nmos))
        for i in range(nmos):
            psi_i = np.abs(np.dot(self._ao, calc_o.mo_coeff[:, i]))
            for j in range(nmos):
                psi_j = np.abs(np.dot(self._ao, calc_d.mo_coeff[:, j]))
                prop = psi_i * psi_j
                sim[i, j] = np.sum(prop * self._grid.weights)

        row, col = sco.linear_sum_assignment(sim, maximize=True)
        scores = sim[row, col]

        if min(scores) < 0.9:
            center = (origin + dest) / 2
            mapping_left, _, calc_c = self._connect(origin, center, calc_o=calc_o)
            mapping_right, _, _ = self._connect(
                center, dest, calc_o=calc_c, calc_d=calc_d
            )
            return mapping_right[mapping_left], calc_o, calc_d
        else:
            return col, calc_o, calc_d

    def __init__(self, origin, destination):
        gridmol = self._get_mol()
        grid = pyscf.dft.gen_grid.Grids(gridmol)
        grid.level = 3
        grid.build()
        self._grid = grid
        self._ao = pyscf.dft.numint.eval_ao(gridmol, grid.coords, deriv=0)

        self._meta_origin = origin
        self._meta_destination = destination
        self._meta_direction = self._meta_destination - self._meta_origin
        self._calcs = {}
        mapping, _, _ = self._connect(0, 1)
        return mapping


o = np.array((7, 5, 7, 5, 6, 6))
d = np.array((7, 5, 5, 7, 6, 6))
c = ConnectedBenzene(o, d)
#%%
# only calculate similarity between elements within energy window
