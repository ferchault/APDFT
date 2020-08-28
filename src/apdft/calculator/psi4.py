#!/usr/bin/env python

import basis_set_exchange as bse
import apdft.calculator as apc
import os
import jinja2 as j
import numpy as np
from apdft import log
import functools


class Psi4Calculator(apc.Calculator):
    _methods = {"PBE": "pbe"}

    @staticmethod
    def _format_atoms(nuclear_numbers, nuclear_charges, coordinates):
        atoms = []
        totalghostcharge = 0
        pointcharges = []
        for atom in range(len(nuclear_charges)):
            linestr = ""
            if nuclear_charges[atom] != float(nuclear_numbers[atom]):
                # Treat as ghost atom
                linestr = "@"
                totalghostcharge -= nuclear_numbers[atom]
                ptchg = f"Chrgfield.extern.addCharge({nuclear_charges[atom]}, {coordinates[atom][0]},{coordinates[atom][1]},{coordinates[atom][2]})"
                pointcharges.append(ptchg)
            linestr += f"{nuclear_numbers[atom]} {coordinates[atom][0]} {coordinates[atom][1]} {coordinates[atom][2]}"
            atoms.append(linestr)
        return "\n".join(atoms), totalghostcharge, "\n".join(pointcharges)

    @staticmethod
    def _format_basis(nuclear_numbers, basisset):
        elements = list(set(nuclear_numbers))
        # workaround for https://github.com/psi4/psi4/issues/1996
        basisstr = bse.get_basis(basisset, fmt="psi4", header=False, elements=elements)
        return basisstr.replace("D+", "E+")

    def get_input(
        self,
        coordinates,
        nuclear_numbers,
        nuclear_charges,
        grid,
        iscomparison=False,
        includeonly=None,
    ):
        basedir = os.path.dirname(os.path.abspath(__file__))
        with open("%s/templates/psi4.txt" % basedir) as fh:
            template = j.Template(fh.read())

        env = {}
        env["ghostedatoms"], env["totalghostcharge"], env[
            "pointcharges"
        ] = Psi4Calculator._format_atoms(nuclear_numbers, nuclear_charges, coordinates)
        env["basisset"] = Psi4Calculator._format_basis(nuclear_numbers, self._basisset)
        env["method"] = self._methods[self._method]

        return template.render(**env)

    @staticmethod
    def get_total_energy(folder):
        with open(f"{folder}/run.inp.dat") as fh:
            lines = fh.readlines()
        for line in lines:
            if "    Total Energy = " in line:
                return float(line.strip().split()[-1])
        raise ValueError("No energy available.")

    def get_runfile(self, coordinates, nuclear_numbers, nuclear_charges, grid):
        basedir = os.path.dirname(os.path.abspath(__file__))
        with open("%s/templates/psi4-run.sh" % basedir) as fh:
            template = j.Template(fh.read())
        return template.render()

    @staticmethod
    def get_epn(folder, coordinates, includeatoms, nuclear_charges):
        epns = {}
        with open(f"") as fh:
            started = False
            seen_bars = 0
            for line in fh:
                if "Electrostatic Potential (a.u.)" in line:
                    started = True
                    continue
                if started:
                    if "-------------------------------" in line:
                        seen_bars += 1
                    else:
                        parts = line.strip().split()
                        siteid = int(parts[0]) - 1
                        epn = -float(parts[-1])
                    if seen_bars > 1:
                        break

        if len(epns.flatten()) == 0:
            raise ValueError("Incomplete calculation.")

        # check that all included sites are in fact present
        included_results = epns[:, 0].astype(np.int)
        if not set(included_results) == set(includeatoms):
            log.log(
                "Atom selections do not match. Likely the configuration has changed in the meantime.",
                level="error",
            )

        included_results = list(included_results)
        return epns[[included_results.index(_) for _ in includeatoms], 1]

