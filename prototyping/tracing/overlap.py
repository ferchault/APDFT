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
import pandas as pd
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
        start = time.time()
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
        print("spc done", time.time() - start)
        return calc

    def _connect(self, origin, dest, calc_o=None, calc_d=None, psi_o=None, psi_d=None):
        print(origin, dest)
        if calc_o is None:
            calc_o = self._do_run(origin)
        if calc_d is None:
            calc_d = self._do_run(dest)

        nmos = len(calc_o.mo_occ)
        sim = np.zeros((nmos, nmos))
        if psi_o is None:
            psi_o = [
                np.abs(np.dot(self._ao, calc_o.mo_coeff[:, i])) for i in range(nmos)
            ]
        if psi_d is None:
            psi_d = [
                np.abs(np.dot(self._ao, calc_d.mo_coeff[:, j])) for j in range(nmos)
            ]
        for i in range(nmos):
            for j in range(nmos):
                sim[i, j] = np.sum(psi_o[i] * psi_d[j] * self._grid.weights)

        row, col = sco.linear_sum_assignment(sim, maximize=True)
        self._sims[(origin, dest)] = sim.copy()
        scores = sim[row, col]

        if min(scores) < 0.7:
            center = (origin + dest) / 2
            mapping_left, _, calc_c, _, psi_c = self._connect(
                origin, center, calc_o=calc_o, psi_o=psi_o
            )
            mapping_right, _, _, _, _ = self._connect(
                center, dest, calc_o=calc_c, calc_d=calc_d, psi_o=psi_c, psi_d=psi_d
            )
            return mapping_right[mapping_left], calc_o, calc_d, psi_o, psi_d
        else:
            self._mappings[(origin, dest)] = col
            return col, calc_o, calc_d, psi_o, psi_d

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
        self._mappings = {}
        self._sims = {}

    def connect(self):
        mapping, _, _, _, _ = self._connect(0, 1)
        return mapping


o = np.array((5, 7, 6, 6, 6, 6))
d = np.array((5, 6, 7, 6, 6, 6))
c = ConnectedBenzene(o, d)
c.connect()
#%%
# only calculate similarity between elements within energy window

# %%
def rank_plot(c):
    f = plt.figure(figsize=(8, 12))
    occupied = sum(c._calcs[0].mo_occ > 0)
    plt.axhline(occupied + 0.5, color="red")

    # get lvals with a change
    lvals = sorted(c._calcs.keys())
    svals = []
    smaps = []
    nmos = len(c._calcs[0].mo_energy)
    for origin, destination in zip(lvals[:-1], lvals[1:]):
        if len(svals) == 0:
            svals.append(origin)
        if np.allclose(c._mappings[(origin, destination)], np.arange(nmos)):
            continue
        svals.append(destination)
        smaps.append(c._mappings[(origin, destination)])
    if svals[-1] < 1:
        svals[-1] = 1

    # plot lvals
    xpos = np.linspace(0, 1, len(svals))
    for x, sval in zip(xpos, svals):
        try:
            coloridx = list(sval % (1 / 2 ** np.arange(10))).index(0.0)
        except:
            coloridx = 10
        plt.scatter(np.zeros(nmos) + x, np.arange(nmos), color=f"C{coloridx}")

    # plot connections
    for idx in range(len(svals) - 1):
        for f, t in enumerate(smaps[idx]):
            if f == t:
                alpha = 0.7
            else:
                alpha = 1
            plt.plot(xpos[idx : idx + 2], (f, t), color="grey", alpha=alpha, zorder=-10)

    # label
    plt.xlabel("Mixing parameter $\lambda$ [non-linear spacing]")
    plt.xticks(xpos, svals)
    plt.ylabel("MO index")


#%%
def extract(c, storename):
    store = pd.HDFStore(f"{storename}.h5")

    calcs = []
    for k, calc in c._calcs.items():
        row = {
            "identifier": k,
            "mo_coeff": calc.mo_coeff,
            "mo_occ": calc.mo_occ,
            "mo_energy": calc.mo_energy,
            "energy": calc.e_tot,
            "origin": c._meta_origin,
            "destination": c._meta_destination,
        }
        calcs.append(row)
    store["calcs"] = pd.DataFrame(calcs)

    maps = []
    for k, map in c._mappings.items():
        maps.append({"identifier": k, "map": map})
    store["mappings"] = pd.DataFrame(maps)

    sims = []
    for k, sim in c._sims.items():
        sims.append({"identifier": k, "sim": sim})
    store["sims"] = pd.DataFrame(sims)

    store.close()


#%%
import sys

origin, destination = sys.argv[1:]
origin = np.array([{"C": 6, "B": 5, "N": 7}[_] for _ in origin])
destination = np.array([{"C": 6, "B": 5, "N": 7}[_] for _ in destination])

c = ConnectedBenzene(origin, destination)
c.connect()
extract(c, f"{sys.argv[1]}-{sys.argv[2]}")
#%%
