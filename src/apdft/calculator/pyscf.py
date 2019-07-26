#!/usr/bin/env python

import apdft.calculator as apc
import os

class PyscfCalculator(apc.Calculator):
    _methods = {"CCSD": "ccsd"}

    def get_input(
        self, coordinates, nuclear_numbers, nuclear_charges, grid, iscomparison=False
    ):
        basedir = os.path.dirname(os.path.abspath(__file__))
        with open("%s/templates/mrcc.txt" % basedir) as fh:
            template = j.Template(fh.read())

        env_coord = GaussianCalculator._format_coordinates(nuclear_numbers, coordinates)
        env_basis = self._basisset
        env_numatoms = len(nuclear_numbers)
        env_charged = MrccCalculator._format_charges(
            coordinates, nuclear_numbers, nuclear_charges
        )

        return template.render(
            coordinates=env_coord,
            method=self._methods[self._method],
            basisset=env_basis,
            numatoms=env_numatoms,
            charges=env_charged,
        )