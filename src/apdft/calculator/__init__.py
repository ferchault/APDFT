#!/usr/bin/env python

import random
import os
import string

import jinja2 as j


class CalculatorInterface(object):
    """ All the functions that need to be implemented for a new code to be supported in APDFT. """

    def get_input(
        self,
        coordinates,
        nuclear_numbers,
        nuclear_charges,
        grid,
        iscomparison=False,
        includeonly=None,
    ):
        """ Generates the calculation input file for the external code. 
        
        It is crucial to remember that this may involve calculations where the basis set of a nucleus and its element to not match.

        Args:
            coordinates:		A (3,N) array of nuclear coordinates :math:`\\mathbf{r_i}`. [Angstrom]
			nuclear_numbers:	A N array of nuclear numbers :math:`q_i`. [e]
            nuclear_charges:	A N array of the effective nuclear charges :math:`q_i`. [e]
            grid:               Deprecated.
            iscomparison:       Boolean. If the input is meant for a comparison calculation, this might allow for shortcuts.
            includeonly:        A N' array of 0-based atom indices to be included in the evaluation.
        Returns:
            File contents as string.
        """
        raise NotImplementedError()

    @staticmethod
    def get_total_energy(folder):
        """ Extracts the total energy of a calculation.
        
        Args:
            folder:             String. Path to the QM calculation from which the energy is to be extracted.
        Returns:
            Total energy including nuclear-nuclear interaction [Hartree]."""
        raise NotImplementedError()

    def get_runfile(self, coordinates, nuclear_numbers, nuclear_charges, grid):
        raise NotImplementedError()

    def get_epn(folder, coordinates, includeatoms, nuclear_charges):
        """ Extracts the electronic contribution to the electrostatic potential at the nuclei. """
        raise NotImplementedError()


class Calculator(CalculatorInterface):
    """ A concurrency-safe blocking interface for an external QM code."""

    _methods = {}

    def __init__(self, method, basisset, superimpose=False):
        self._method = method
        self._basisset = basisset
        self._superimpose = superimpose

    def get_methods(self):
        return list(self._methods.keys())

    def get_density_on_grid(self, folder, gridpoints):
        raise NotImplementedError()

    @staticmethod
    def _get_tempname():
        return "apdft-tmp-" + "".join(
            random.choice(string.ascii_uppercase + string.digits) for _ in range(10)
        )

    @staticmethod
    def get_grid(nuclear_numbers, coordinates, outputfolder):
        """ Returns the integration grid used by this calculator for a given set of nuclei and geometry.

		Grid weights and coordinates may be in internal units. Return value should be coords, weights. If return value is None, a default grid is used."""
        return None


class MockCalculator(Calculator):
    _methods = {}

    @classmethod
    def get_runfile(self, coordinates, nuclear_numbers, nuclear_charges, grid):
        basedir = os.path.dirname(os.path.abspath(__file__))
        with open("%s/templates/mock-run.sh" % basedir) as fh:
            template = j.Template(fh.read())
        return template.render()
