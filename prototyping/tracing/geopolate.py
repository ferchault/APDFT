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

basepath = "/mnt/c/Users/guido/data/qm9-isomers"
# endregion

# region
class Interpolate:
    def __init__(self, fn_origin, fn_destination, HOMOminus, LUMOplus):
        self._origin = ase.io.read(fn_origin)
        self._destination = ase.io.read(fn_destination)
        self._elements = np.array(self._origin.get_chemical_symbols())
        self._align()

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

    def _do_run(self, lval):
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

        # self._calcs[lval] = calc
        return calc

    def _connect(self, origin, dest):
        pass

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
        self._geometry = sci.interp1d(np.linspace(0, 1, nimages), coords, axis=0)

    def connect(self):
        self._parametrize_geometry()
        # self._connect(0, 1)


i = Interpolate(f"{basepath}/ci0001.xyz", f"{basepath}/ci0002.xyz", 3, 3)
i.connect()
# endregion
# region

# region
import tqdm

calcs = []
for lval in tqdm.tqdm(np.linspace(0, 1, 100)):
    calcs.append(i._do_run(lval))
# region
LUMO = list(calcs[0].mo_occ).index(0)
import matplotlib.pyplot as plt

for shift in range(-5, 5):
    plt.plot([_.mo_energy[LUMO - shift] for _ in calcs])
# region
