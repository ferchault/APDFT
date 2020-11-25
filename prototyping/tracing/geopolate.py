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
from ase.optimize.fire import FIRE as QuasiNewton
from xtb.ase.calculator import XTB

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
        if fractionalval[1] > 2**15:
            raise ValueError("Steps too tiny. Find some other path.")
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
        percent = origin[0] / origin[1] * 100
        print(f"{percent:6.2f} %: {origin[0]}/{origin[1]} to {dest[0]}/{dest[1]}")
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
        nimages = 20
        images = [self._origin]
        images += [self._origin.copy() for i in range(nimages - 2)]
        images += [self._destination]
        neb = NEB(images)
        neb.interpolate("idpp")
        #for image in images:
        #    image.calc = XTB()
        #f = QuasiNewton(neb)
        #f.run(steps=1)

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
