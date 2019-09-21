#!/usr/bin/env python

import os
import numpy as np
import basis_set_exchange as bse
import jinja2 as j

from apdft import calculator
from apdft import log
import apdft.physics as physics
from apdft.calculator.gaussian import GaussianCalculator


class MrccCalculator(calculator.Calculator):
    _methods = {"CCSDT": "ccsdt", "CCSD": "ccsd", "HF": "RHF"}

    @staticmethod
    def _parse_densityfile(densityfile):
        """ Returns all relevant data from a MRCC density file.

		Columns 0-2: x, y, z coordinates in bohr
		Column 3: weights
		Column 4: density"""
        with open(densityfile, "r") as fh:
            _ = np.fromfile(fh, "i4")
            q = _[3:-1].view(np.float64)
            ccdensity = q.reshape((-1, 10))
        return ccdensity[:, 1:6]

    @staticmethod
    def density_on_grid(densityfile, grid):
        ccdensity = MrccCalculator._parse_densityfile(densityfile)
        if not np.allclose(grid, ccdensity[:, :3] / physics.angstrom):
            raise ValueError("Unable to combine different grids.")
        return ccdensity[:, 4]

    @staticmethod
    def get_grid(nuclear_numbers, coordinates, outputfolder):
        """ Obtains the integration grid from one of the MRCC output files.

        Returns:
            Grid coordinates [Angstrom], grid weights."""
        ccdensity = MrccCalculator._parse_densityfile("%s/DENSITY" % outputfolder)
        return ccdensity[:, :3] / physics.angstrom, ccdensity[:, 3]

    @staticmethod
    def _format_charges(coordinates, nuclear_numbers, nuclear_charges):
        ret = []
        for coord, Z_ref, Z_tar in zip(coordinates, nuclear_numbers, nuclear_charges):
            ret.append("%f %f %f %f" % (coord[0], coord[1], coord[2], (Z_tar - Z_ref)))
        return "\n".join(ret)

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

    @classmethod
    def get_runfile(self, coordinates, nuclear_numbers, nuclear_charges, grid):
        basedir = os.path.dirname(os.path.abspath(__file__))
        with open("%s/templates/mrcc-run.sh" % basedir) as fh:
            template = j.Template(fh.read())
        return template.render()

    def get_density_on_grid(self, folder, grid):
        return MrccCalculator.density_on_grid(folder + "/DENSITY", grid)

    @staticmethod
    def get_total_energy(folder):
        """ Returns the total energy in Hartree."""
        logfile = "%s/run.log" % folder
        energy = None
        with open(logfile) as fh:
            lines = fh.readlines()[::-1]
            for line in lines:
                if "Total CCSD energy [au]:" in line:
                    energy = float(line.strip().split()[-1])
                    break
                if "***FINAL KOHN-SHAM ENERGY:" in line:
                    energy = float(line.strip().split()[-2])
                    break
        if energy is None:
            log.log(
                "Unable to read energy from log file.", filename=logfile, level="error"
            )
            return 0
        return energy

    @staticmethod
    def get_electronic_dipole(folder, gridcoords, gridweights):
        raise NotImplementedError()

    @staticmethod
    def get_epn(folder, coordinates, includeatoms, nuclear_charges):
        """ Calculates the EPN from a MRCC density file.

        MRCC has no support to evaluate the EPN directly. To avoid parsing the ordering of the density matrix in ao basis, the density on a grid is read instead. The resulting error should be small for the first few orders.
            
        Args:
            folder:         String, the path to the calculation.
            coordiantes:    Nuclear coordinates
            includeatoms:   List of zero-based indices of the atoms to include in APDFT
            nuclear_charges: Float, list of nuclear charges for this particular calculation.
        Returns:
            Numpy array of EPN in Hartree."""

        components = MrccCalculator._parse_densityfile("%s/DENSITY" % folder)
        gridpos, gridweights, density = (
            components[:, :3],
            components[:, 3],
            components[:, 4],
        )

        epns = []
        for atomidx in includeatoms:
            ds = np.linalg.norm(
                gridpos - coordinates[atomidx] * physics.angstrom, axis=1
            )
            epns.append((gridweights * density / ds).sum())
        return np.array(epns)
