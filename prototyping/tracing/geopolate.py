#!/usr/bin/env python
# region
import numpy as np
import ase
import sys
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
    def __init__(self, fn_origin, fn_destination):
        self._origin = ase.io.read(fn_origin)
        self._destination = ase.io.read(fn_destination)
        self._elements = np.array(self._origin.get_chemical_symbols())
        self._align()
        self._calcs = {}
        self._sims = {}
        self._parametrize_geometry()

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

    def _do_run(self, fractionalval, mo_coeff=None, mo_occ=None):
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

        # debug
        # with open(f"{fractionalval[0]}-{fractionalval[1]}.xyz", "w") as fh:
        #    atom = "\n".join(atom)
        #    fh.write(f"{19}\n\n{atom}\n")

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

        dm = None
        calc.chkfile = None
        if mo_coeff is not None:
            dm = calc.make_rdm1(mo_coeff, mo_occ)
        hfe = calc.kernel(dm, verbose=5)
        if not calc.converged:
            print(atom)
            raise ValueError("unconverged")

        self._calcs[fractionalval] = calc
        self._nmos = len(calc.mo_energy)
        return calc

    def _connect(self, origin, dest, calc_o=None, calc_d=None):
        if origin[1] != dest[1]:
            raise NotImplementedError("logic error!")
        percent = int(origin[0] / origin[1] * 100)
        print("<->", origin, dest, f"{percent}%")
        mo_coeff, mo_occ = None, None
        if calc_o is None:
            if calc_d is not None:
                mo_coeff = calc_d.mo_coeff
                mo_occ = calc_d.mo_occ
            try:
                calc_o = self._do_run(origin)
            except:
                calc_o = self._do_run(origin, mo_coeff, mo_occ)
        if calc_d is None:
            if calc_o is not None:
                mo_coeff = calc_o.mo_coeff
                mo_occ = calc_o.mo_occ
            try:
                calc_d = self._do_run(dest)
            except:
                calc_d = self._do_run(dest, mo_coeff, mo_occ)

        s = pyscf.gto.intor_cross("int1e_ovlp", calc_o.mol, calc_d.mol)
        sim = np.abs(np.dot(np.dot(calc_o.mo_coeff.T, s), calc_d.mo_coeff))

        row, col = sco.linear_sum_assignment(sim, maximize=True)
        scores = sim[row, col][18:48]
        self._sims[(origin, dest)] = sim.copy()

        print("    ", min(scores))
        if 0.0 in scores:
            print("   MO", list(scores).index(0.0) + 18)
        if min(scores) < 0.7:
            center = (origin[0] * 2 + 1, origin[1] * 2)
            calc_o, calc_c = self._connect(
                (origin[0] * 2, origin[1] * 2), center, calc_o=calc_o
            )
            self._connect(
                center, (dest[0] * 2, dest[1] * 2), calc_o=calc_c, calc_d=calc_d
            )
        return calc_o, calc_d

    def _parametrize_geometry(self):
        nimages = 7
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
            np.linspace(0, 1, nimages), coords, axis=0, kind="linear"
        )

    def connect(self):
        self._connect((0, 1), (1, 1))

    def save(self, fn):
        # scale exact positions
        max_denominator = max([_[1] for _ in self._calcs.keys()])

        # calculations
        rows = []
        for pos, calc in self._calcs.items():
            factor = max_denominator // pos[1]
            rows.append(
                {"pos": pos[0] * factor, "occ": calc.mo_occ, "energies": calc.mo_energy}
            )
        calcs = pd.DataFrame(rows).sort_values("pos").reset_index(drop=True)

        # similarities
        rows = []
        lvals = sorted(self._calcs.keys(), key=lambda _: _[0] / _[1])
        for origin, destination in zip(lvals[:-1], lvals[1:]):
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
            sim = self._sims[identifier]
            rows.append(
                {
                    "origin": origin[0] * max_denominator // origin[1],
                    "destination": destination[0] * max_denominator // destination[1],
                    "sim": sim,
                }
            )

        sims = pd.DataFrame(rows)

        # store
        with pd.HDFStore(fn) as store:
            store["calcs"] = calcs
            store["sims"] = sims

    def write_path(self, filename, nsteps=100):
        with open(filename, "w") as fh:
            natoms = len(self._elements)
            for x in np.linspace(0, 1, nsteps):
                atom = []
                for element, position in zip(self._elements, self._geometry(x)):
                    atom.append(f"{element} {position[0]} {position[1]} {position[2]}")
                atom = "\n".join(atom)
                fh.write(f"{natoms}\n\n{atom}\n")


if __name__ == "__main__":
    fnA, fnB, fnout = sys.argv[1:]
    i = Interpolate(fnA, fnB)
    i.write_path(f"{fnout}.xyz")
    i.connect()
    i.save(f"{fnout}.h5")
# endregion
# region

# # region
# import tqdm

# calcs = []
# for lval in tqdm.tqdm(np.linspace(0.0, 1, 64)):
#     calcs.append(i._do_run(lval))
# # region
# LUMO = list(calcs[0].mo_occ).index(0)
# import matplotlib.pyplot as plt

# for shift in range(1, 5):
#     plt.plot([_.mo_energy[LUMO - shift] for _ in calcs][0:10], "o-")
# # region
# with open("debug.xyz", "w") as fh:
#     for x in np.linspace(0, 1, 100):
#         atom = []
#         for element, position in zip(i._elements, i._geometry(x)):
#             atom.append(f"{element} {position[0]} {position[1]} {position[2]}")
#         fh.write("19\n\n" + "\n".join(atom) + "\n")
# # region
# def overlap(calc1, calc2, id1, id2):
#     grid = pyscf.dft.gen_grid.Grids(calc1.mol)
#     grid.level = 3
#     grid.build()
#     ao = pyscf.dft.numint.eval_ao(calc1.mol, grid.coords, deriv=0)
#     psi1 = np.dot(ao, calc1.mo_coeff[:, id1])
#     psi2 = np.dot(ao, calc2.mo_coeff[:, id2])
#     return np.sum(np.abs(psi1 * psi2) * grid.weights)


# for other in (2, 3, 4):
#     q = overlap(calcs[1], calcs[50], LUMO - 3, LUMO - other)
#     print(other, q)
# # region
# calcs[0].mol
# # region

# # region
# for shift in range(1, 5):
#     lvals = sorted(i._calcs.keys())
#     plt.plot(lvals, [i._calcs[lval].mo_energy[LUMO - shift] for lval in lvals])
# # region
# def follow_me(gp):
#     lvals = sorted(gp._calcs.keys(), key=lambda _: _[0] / _[1])
#     nmos = gp._nmos
#     labels = np.array([f"MO-{_}" for _ in range(nmos)])
#     positions = []
#     for idx, destination in enumerate(lvals):
#         if idx == 0:
#             ranking = labels
#         else:
#             origin = lvals[idx - 1]
#             print(origin, destination)
#             maxq = max(origin[1], destination[1])
#             originfactor = maxq // origin[1]
#             destinationfactor = maxq // destination[1]
#             identifier = (
#                 (origin[0] * originfactor, origin[1] * originfactor),
#                 (
#                     destination[0] * destinationfactor,
#                     destination[1] * destinationfactor,
#                 ),
#             )
#             sim = gp._sims[identifier]
#             row, col = sco.linear_sum_assignment(sim, maximize=True)
#             ranking = positions[-1][np.argsort(col)]
#         positions.append(ranking)
#     return positions
# region
mol = pyscf.gto.Mole()
mol.atom = """C -1.097745727159791 0.23485654734900535 -0.9088858026239317
C -1.148721294308506 0.631582894648981 0.37109864730002834
C -1.3035513191474692 -0.6139636659989773 -0.101941083091326
O 0.1276146939024135 1.0352450310600259 1.1278270963572063
C 0.11389400908899736 -0.008765093638428461 0.01884833161651299
C 1.2863944794215965 0.45568563318279715 0.8086357123932881
C 2.4055931469398595 -0.00884017654152008 0.6024516548181206
C 2.291886858183722 -0.7891541441296157 -0.25052046816797835
O 1.0581018481417324 -0.8588218652550573 -0.6379288725084366
H -2.1672358736281283 0.8465780284891521 -1.2354890398436469
H -0.9435696858430008 1.4053112366460676 -1.3645788377943546
H -1.117748460709315 0.11217639495404938 -2.1276487852886214
H -2.5361255510934653 -0.6807044908125485 0.21453281938619664
H -1.5776349717434666 -1.787727326679819 -0.32134326417817033
H -1.552802796015216 -1.1177728598481522 1.031610529504163
H -1.7224031225089327 1.460433354191268 1.0967697482294911
H 1.5056349897114922 1.3385120499258043 1.7993573331553971
H 3.6049585143847995 -0.02438031611837757 0.8925596162660016
H 2.7734602623826796 -1.6302512314246582 -1.0153553355299398"""
mol.basis = "6-31G"
mol.verbose = 4
mol.build()

q = ase.io.read("fail.xyz")
deltaZ = q.get_atomic_numbers() - q.get_atomic_numbers()
print(deltaZ)


def add_qmmm(calc, mol, deltaZ):
    mf = pyscf.qmmm.mm_charge(calc, mol.atom_coords() * nist.BOHR, deltaZ)

    def energy_nuc(self):
        q = mol.atom_charges().astype(np.float)
        q += deltaZ
        return mol.energy_nuc(q)

    mf.energy_nuc = energy_nuc.__get__(mf, mf.__class__)

    return mf


calc = add_qmmm(pyscf.scf.RHF(mol), mol, deltaZ)

dm = None
hfe = calc.kernel(dm, verbose=5)
#%%
i = Interpolate("ci0001.xyz", "ci0100.xyz")
# region
i._connect((0, 1), (1, 1))
# region
calc1 = i._calcs[(127, 256)]
calc2 = i._calcs[(1, 2)]
s = pyscf.gto.intor_cross("int1e_ovlp", calc1.mol, calc2.mol)
sim = np.abs(np.dot(np.dot(calc1.mo_coeff.T, s), calc2.mo_coeff))
row, col = sco.linear_sum_assignment(sim, maximize=True)
plt.plot(sim[row, col])
plt.ylim(0, 1)
plt.axhline(0.7)
# region
127 * 2
# region
i._do_run((255, 512))
# region
calc1 = i._calcs[(127, 256)]
calc2 = i._calcs[(1, 2)]
plt.plot(calc1.mo_energy - calc2.mo_energy)
# region
i._calcs.keys()
# region
254 / 2
# region
lvals = sorted(i._calcs.keys(), key=lambda _: _[0] / _[1])
xs = [_[0] / _[1] for _ in lvals]
ys = np.array([i._calcs[_].mo_energy for _ in lvals])
# region
plt.plot(xs, ys[:, 34])
plt.plot(xs, ys[:, 33])
plt.plot(xs, ys[:, 32])
plt.plot(xs, ys[:, 31])
# region
mycas = calc1.CASCI(6, 8).run()
# region
mycas.mo_coeff - calc1.mo_coeff
# region
xs = np.arange(120, 135)
for x in xs:
    i._do_run((x, 256))
# region
hfcalcs = [i._calcs[(x, 256)] for x in xs]
# cascalcs = [_.CASCI(6, 8).run() for _ in hfcalcs]
# region
ys = np.array([_.mo_energy for _ in hfcalcs])
for idx in range(32, 38):
    plt.plot(xs, ys[:, idx], "o-")
# region
n = i._calcs[(1, 2)]
i._do_run((7, 16), mo_coeff=n.mo_coeff, mo_occ=n.mo_occ)
# region
