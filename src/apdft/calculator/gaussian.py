#!/usr/bin/env python
from apdft import log
from apdft import calculator
import apdft.physics as ap

import os
import sys
import numpy as np
import basis_set_exchange as bse
import jinja2 as j


class GaussianCalculator(calculator.Calculator):
    """ Performs the QM calculations for APDFT with the help of Gaussian.

    General Idea:
    Gaussian supports both fractional nuclar charges (via the undocumented `Massage` keyword) and the evaluation of the electrostatic potential at the nucleus (via the `Prop` keyword). Earlier versions used to read fchk files and map out a grid, which is less accurate, but also feasible in Gaussian."""

    _methods = {
        "CCSD": "CCSD(Full,MaxCyc=100)",
        "CCSDT": "CCSDT(Full,MaxCyc=100)",
        "PBE0": "PBE1PBE",
        "PBE": "PBEPBE",
        "HF": "UHF",
        "HSE06": "HSEH1PBE",
        "B3LYP": "B3LYP",
        "M06L": "M06L integral=ultrafine",
        "TPSS": "TPSSTPSS",
    }

    @staticmethod
    def _format_coordinates(nuclear_numbers, coordinates):
        ret = ""
        for Z, coords in zip(nuclear_numbers, coordinates):
            ret += "%d %f %f %f\n" % (Z, coords[0], coords[1], coords[2])
        return ret[:-1]

    @staticmethod
    def _format_basisset(nuclear_charges, basisset, superimposed=False):
        res = ""
        for atomid, nuclear_charge in enumerate(nuclear_charges):
            if superimposed:
                elements = set(
                    [
                        max(1, int(_(nuclear_charge)))
                        for _ in (
                            np.round,
                            lambda _: np.round(_ + 1),
                            lambda _: np.round(_ - 1),
                        )
                    ]
                )
            else:
                elements = set([max(1, int(_(nuclear_charge))) for _ in (np.round,)])
            output = bse.get_basis(basisset, elements=list(elements), fmt="gaussian94")

            res += "%d 0\n" % (atomid + 1)
            skipnext = False
            for line in output.split("\n"):
                if line.startswith("!"):
                    skipnext = False
                    continue
                if len(line.strip()) == 0 or line.strip() == "****":
                    skipnext = True
                    continue
                if skipnext:
                    skipnext = False
                    continue
                res += line + "\n"
            res += "****\n"

        return res.strip()

    @staticmethod
    def _format_nuclear(nuclear_charges):
        return "\n".join(
            ["%d Nuc %f" % (_[0] + 1, _[1]) for _ in enumerate(nuclear_charges)]
        )

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
        with open("%s/templates/gaussian.txt" % basedir) as fh:
            template = j.Template(fh.read())

        env_coord = GaussianCalculator._format_coordinates(nuclear_numbers, coordinates)
        env_basis = GaussianCalculator._format_basisset(
            nuclear_charges, self._basisset, self._superimpose
        )
        env_nuc = GaussianCalculator._format_nuclear(nuclear_charges)
        env_molcharge = int(np.sum(nuclear_charges) - np.sum(nuclear_numbers))
        return template.render(
            coordinates=env_coord,
            method=self._methods[self._method],
            basisset=env_basis,
            nuclearcharges=env_nuc,
            moleculecharge=env_molcharge,
        )

    @classmethod
    def get_runfile(self, coordinates, nuclear_numbers, nuclear_charges, grid):
        basedir = os.path.dirname(os.path.abspath(__file__))
        with open("%s/templates/gaussian-run.sh" % basedir) as fh:
            template = j.Template(fh.read())
        return template.render()

    @staticmethod
    def get_total_energy(folder):
        """ Returns the total energy in Hartree."""
        chkfile = "%s/run.fchk" % folder
        with open(chkfile) as fh:
            lines = fh.readlines()
        energy = None
        scflines = [_ for _ in lines if _.startswith("Total Energy")]
        if len(scflines) > 0:
            energy = float(scflines[0].strip().split()[-1])
        return energy

    @staticmethod
    def get_epn(folder, coordinates, includeatoms, nuclear_charges):
        """ Extracts the EPN from a Gaussian log file. 

        The Gaussian convention is to include the nuclear interaction of all other sites. Signs are in the physical sense, i.e. the electronic contribution is negative, while the nuclear contribution is positive. This function also converts to the APDFT convention where :math:`\\int \\rho / |\\mathbf{r}-\\mathbf{R}_I|` is positive. 
            
        Args:
            folder:         String, the path to the calculation.
            coordiantes:    Nuclear coordinates
            includeatoms:   List of zero-based indices of the atoms to include in APDFT
            nuclear_charges: Float, list of nuclear charges for this particular calculation.
        Returns:
            Numpy array of EPN in Hartree."""
        with open(folder + "/run.log") as fh:
            lines = fh.readlines()

        offset = (
            lines.index(
                "    Center     Electric         -------- Electric Field --------\n"
            )
            + 3
        )
        epns = []
        for atomidx, line in enumerate(lines[offset : offset + len(coordinates)]):
            if atomidx not in includeatoms:
                continue
            epn = float(line.strip().split()[2])
            epn -= ap.Coulomb.nuclear_potential(coordinates, nuclear_charges, atomidx)
            epns.append(-epn)
        return np.array(epns)
