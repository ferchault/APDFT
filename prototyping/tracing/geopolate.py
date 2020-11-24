#!/usr/bin/env python
# region
import numpy as np
import ase
from ase.neb import NEB
import rmsd
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
import scipy.interpolate as sci
from ase.calculators.emt import EMT
from ase.optimize.fire import FIRE as QuasiNewton

basepath = "."
# endregion

# region
class Interpolate:
    def __init__(self, fn_origin, fn_destination, LUMOminus, LUMOplus):
        self._origin = ase.io.read(fn_origin)
        self._destination = ase.io.read(fn_destination)
        self._elements = np.array(self._origin.get_chemical_symbols())
        self._align()
        self._MOrange = (LUMOminus, LUMOplus + 1)
        self._calcs = {}
        self._sims = {}

    def _align(self):
        A = self._origin.get_positions()
        B = self._destination.get_positions()

        # remove shift
        A -= rmsd.centroid(A)
        B -= rmsd.centroid(B)

        # remove rotation
        U = rmsd.kabsch(A, B)
        A = np.dot(A, U)

        # reorder
        elements_destination = np.array(self._destination.get_chemical_symbols())
        mapping = rmsd.reorder_hungarian(self._elements, elements_destination, A, B)

        self._origin.set_positions(A)
        self._destination.set_positions(B[mapping])
        self._destination.set_chemical_symbols(elements_destination[mapping])

    def _do_run(self, fractionalval):
        print(f"  > {fractionalval}")
        lval = fractionalval[0] / fractionalval[1]
        # build molecule
        mol = pyscf.gto.Mole()
        atom = []
        for element, position in zip(self._elements, self._geometry(lval)):
            atom.append(f"{element} {position[0]} {position[1]} {position[2]}")
        mol.atom = "\n".join(atom)
        mol.basis = "6-31G"
        mol.verbose = 0
        mol.build()

        deltaZ = lval * (
            self._destination.get_atomic_numbers() - self._origin.get_atomic_numbers()
        )

        def add_qmmm(calc, mol, deltaZ):
            mf = pyscf.qmmm.mm_charge(calc, mol.atom_coords() * nist.BOHR, deltaZ)

            def energy_nuc(self):
                q = mol.atom_charges().astype(np.float)
                q += deltaZ
                return mol.energy_nuc(q)

            mf.energy_nuc = energy_nuc.__get__(mf, mf.__class__)

            return mf

        calc = add_qmmm(pyscf.scf.RHF(mol), mol, deltaZ)
        hfe = calc.kernel(verbose=0)

        self._calcs[fractionalval] = calc
        self._nmos = len(calc.mo_occ)
        self._LUMO = list(calc.mo_occ).index(0)

        grid = pyscf.dft.gen_grid.Grids(calc.mol)
        grid.level = 3
        grid.build()
        calc.ao = pyscf.dft.numint.eval_ao(calc.mol, grid.coords, deriv=0)
        calc.gp = grid.coords
        calc.gw = grid.weights
        return calc

    def _connect(self, origin, dest, calc_o=None, calc_d=None):
        if origin[1] != dest[1]:
            raise NotImplementedError("logic error!")
        print("<->", origin, dest)
        if calc_o is None:
            calc_o = self._do_run(origin)
        if calc_d is None:
            calc_d = self._do_run(dest)

        sim = np.zeros((self._nmos, self._nmos))
        psi_o = [np.dot(calc_o.ao, calc_o.mo_coeff[:, i]) for i in range(self._nmos)]
        psi_d = [np.dot(calc_o.ao, calc_d.mo_coeff[:, i]) for i in range(self._nmos)]
        for i in range(self._nmos):
            for j in range(self._nmos):
                deltaE = abs(calc_o.mo_energy[i] - calc_d.mo_energy[j])
                if deltaE > 1 / 27.2114:
                    sim[i, j] = 0.0
                else:
                    sim[i, j] = np.sum(np.abs(psi_o[i] * psi_d[j]) * calc_o.gw)

        row, col = sco.linear_sum_assignment(sim, maximize=True)
        scores = sim[row, col]
        self._sims[(origin, dest)] = sim.copy()

        if min(scores) < 0.75:
            center = (origin[0] * 2 + 1, origin[1] * 2)
            calc_o, calc_c = self._connect(
                (origin[0] * 2, origin[1] * 2), center, calc_o=calc_o
            )
            self._connect(
                center, (dest[0] * 2, dest[1] * 2), calc_o=calc_c, calc_d=calc_d
            )
        return calc_o, calc_d

    def _parametrize_geometry(self):
        nimages = 5
        images = [self._origin]
        images += [self._origin.copy() for i in range(nimages - 2)]
        images += [self._destination]
        neb = NEB(images)
        neb.interpolate("idpp")
        for image in images:
            image.calc = EMT()
        f = QuasiNewton(neb)
        f.run(steps=50)

        coords = [_.get_positions() for _ in neb.images]
        self._geometry = sci.interp1d(
            np.linspace(0, 1, nimages), coords, axis=0, kind="quadratic"
        )

    def connect(self):
        self._parametrize_geometry()
        self._connect((0, 1), (1, 1))


i = Interpolate(f"{basepath}/ci0001.xyz", f"{basepath}/ci0100.xyz", 3, 3)
i.connect()
# endregion
# region

# region
import tqdm

calcs = []
for lval in tqdm.tqdm(np.linspace(0.0, 1, 64)):
    calcs.append(i._do_run(lval))
# region
LUMO = list(calcs[0].mo_occ).index(0)
import matplotlib.pyplot as plt

for shift in range(1, 5):
    plt.plot([_.mo_energy[LUMO - shift] for _ in calcs][0:10], "o-")
# region
with open("debug.xyz", "w") as fh:
    for x in np.linspace(0, 1, 100):
        atom = []
        for element, position in zip(i._elements, i._geometry(x)):
            atom.append(f"{element} {position[0]} {position[1]} {position[2]}")
        fh.write("19\n\n" + "\n".join(atom) + "\n")
# region
def overlap(calc1, calc2, id1, id2):
    grid = pyscf.dft.gen_grid.Grids(calc1.mol)
    grid.level = 3
    grid.build()
    ao = pyscf.dft.numint.eval_ao(calc1.mol, grid.coords, deriv=0)
    psi1 = np.dot(ao, calc1.mo_coeff[:, id1])
    psi2 = np.dot(ao, calc2.mo_coeff[:, id2])
    return np.sum(np.abs(psi1 * psi2) * grid.weights)


for other in (2, 3, 4):
    q = overlap(calcs[1], calcs[50], LUMO - 3, LUMO - other)
    print(other, q)
# region
calcs[0].mol
# region

# region
for shift in range(1, 5):
    lvals = sorted(i._calcs.keys())
    plt.plot(lvals, [i._calcs[lval].mo_energy[LUMO - shift] for lval in lvals])
# region
def follow_me(gp):
    lvals = sorted(gp._calcs.keys(), key=lambda _: _[0] / _[1])
    nmos = gp._nmos
    labels = np.array([f"MO-{_}" for _ in range(nmos)])
    positions = []
    for idx, destination in enumerate(lvals):
        if idx == 0:
            ranking = labels
        else:
            origin = lvals[idx - 1]
            print(origin, destination)
            maxq = max(origin[1], destination[1])
            originfactor = maxq // origin[1]
            destinationfactor = maxq // destination[1]
            identifier = (
                (origin[0] * originfactor, origin[1] * originfactor),
                (
                    destination[0] * destinationfactor,
                    destination[1] * destinationfactor,
                ),
            )
            sim = gp._sims[identifier]
            row, col = sco.linear_sum_assignment(sim, maximize=True)
            ranking = positions[-1][np.argsort(col)]
        positions.append(ranking)
    return positions


pos = follow_me(i)
# region
import matplotlib.pyplot as plt

for shift in range(1, 6):
    plt.plot(
        sorted(i._calcs.keys(), key=lambda _: _[0] / _[1]),
        [list(_).index(f"MO-{i._LUMO-shift}") for _ in pos],
    )
# region
i._calcs
# region
