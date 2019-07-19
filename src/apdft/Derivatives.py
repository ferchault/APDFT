#!/usr/bin/env python
import itertools as it
import os
import glob
import multiprocessing as mp
import functools
import traceback

import numpy as np
import pandas as pd
import basis_set_exchange as bse
import pyscf
from pyscf import dft

import apdft
import apdft.math
import apdft.physics


class DerivativeFolders(apdft.physics.APDFT):
    def assign_calculator(self, calculator):
        self._calculator = calculator

    @staticmethod
    def _get_target_name(target):
        return ",".join([apdft.physics.charge_to_label(_) for _ in target])

    def _print_energies(self, targets, energies, comparison_energies):
        if comparison_energies is None:
            for target, energy in zip(targets, energies):
                targetname = DerivativeFolders._get_target_name(target)
                apdft.log.log(
                    "Energy calculated",
                    level="RESULT",
                    value=energy,
                    kind="total_energy",
                    target=target,
                    targetname=targetname,
                )
        else:
            for target, energy, comparison in zip(
                targets, energies, comparison_energies
            ):
                targetname = DerivativeFolders._get_target_name(target)
                apdft.log.log(
                    "Energy calculated",
                    level="RESULT",
                    value=energy,
                    kind="total_energy",
                    target=target,
                    targetname=targetname,
                    reference=comparison,
                    error=energy - comparison,
                )

    def _print_dipoles(self, targets, dipoles, comparison_dipoles):
        if comparison_dipoles is not None:
            for target, dipole, comparison in zip(targets, dipoles, comparison_dipoles):
                targetname = DerivativeFolders._get_target_name(target)
                apdft.log.log(
                    "Dipole calculated",
                    level="RESULT",
                    kind="total_dipole",
                    reference=list(comparison),
                    value=list(dipole),
                    target=target,
                    targetname=targetname,
                )

    def _calculate_delta_Z_vector(self, numatoms, order, sites, direction):
        baseline = np.zeros(numatoms)

        if order > 0:
            sign = {"up": 1, "dn": -1}[direction] * self._delta
            baseline[list(sites)] += sign

        return baseline

    def _get_grid(self):
        """ Returns the default integration grid in Angstrom."""
        mol = pyscf.gto.Mole()
        for nuclear, coord in zip(self._nuclear_numbers, self._coordinates):
            # pyscf molecule init is in Angstrom
            mol.atom.extend([[nuclear, *coord]])
        mol.build()
        grid = dft.gen_grid.Grids(mol)
        grid.level = 3
        grid.build()
        # pyscf grid is in a.u.
        return grid.coords / apdft.physics.angstrom, grid.weights

    def update_grid(self):
        """ Loads the integration grid from the calculator or provides a default one."""
        calcgrid = self._calculator.get_grid(
            self._nuclear_numbers, self._coordinates, "QM/order-0/site-all-cc"
        )
        if calcgrid is None:
            self._gridcoords, self._gridweights = self._get_grid()
        else:
            self._gridcoords, self._gridweights = calcgrid

    def prepare(self, explicit_reference=False):
        """ Builds a complete folder list of all relevant calculations."""
        if os.path.isdir("QM"):
            apdft.log.log(
                "Input folder exists. Reusing existing data.", level="warning"
            )
            return

        commands = []

        for order in self._orders:
            # only upper triangle with diagonal
            for combination in it.combinations_with_replacement(
                self._include_atoms, order
            ):
                if len(combination) == 2 and combination[0] == combination[1]:
                    continue
                if order > 0:
                    label = "-" + "-".join(map(str, combination))
                    directions = ["up", "dn"]
                else:
                    directions = ["cc"]
                    label = "-all"

                for direction in directions:
                    path = "QM/order-%d/site%s-%s" % (order, label, direction)
                    os.makedirs(path, exist_ok=True)

                    charges = self._nuclear_numbers + self._calculate_delta_Z_vector(
                        len(self._nuclear_numbers), order, combination, direction
                    )
                    inputfile = self._calculator.get_input(
                        self._coordinates, self._nuclear_numbers, charges, None
                    )
                    with open("%s/run.inp" % path, "w") as fh:
                        fh.write(inputfile)
                    with open("%s/run.sh" % path, "w") as fh:
                        fh.write(
                            self._calculator.get_runfile(
                                self._coordinates, self._nuclear_numbers, charges, None
                            )
                        )
                    commands.append("( cd %s && bash run.sh )" % path)
        if explicit_reference:
            targets = self.enumerate_all_targets()
            apdft.log.log(
                "All targets listed for comparison run.",
                level="info",
                count=len(targets),
            )
            for target in targets:
                path = "QM/comparison-%s" % ("-".join(map(str, target)))
                os.makedirs(path, exist_ok=True)

                inputfile = self._calculator.get_input(
                    self._coordinates, self._nuclear_numbers, target, None
                )
                with open("%s/run.inp" % path, "w") as fh:
                    fh.write(inputfile)
                with open("%s/run.sh" % path, "w") as fh:
                    fh.write(
                        self._calculator.get_runfile(
                            self._coordinates, self._nuclear_numbers, target, None
                        )
                    )
                commands.append("( cd %s && bash run.sh )" % path)

        # write commands
        with open("commands.sh", "w") as fh:
            fh.write("\n".join(commands))

    def _cached_reader(self, folder):
        if folder not in self._reader_cache:
            rho = self._calculator.get_density_on_grid(folder, self._gridcoords)
            self._reader_cache[folder] = rho
            int_electrons = np.sum(rho * self._gridweights)
            num_electrons = sum(self._nuclear_numbers)
            if abs(num_electrons - int_electrons) > 1e-2:
                apdft.log.log(
                    "Electron count mismatch.",
                    level="error",
                    expected=num_electrons,
                    found=int_electrons,
                    path=folder,
                )

        return self._reader_cache[folder]

    def get_density_derivative(self, sites):
        num_electrons = sum(self._nuclear_numbers)
        if len(sites) == 0:
            return self._cached_reader("QM/order-0/site-all-cc")
        if len(sites) == 1:
            rhoup = self._cached_reader("QM/order-1/site-%d-up" % sites[0])
            rhodn = self._cached_reader("QM/order-1/site-%d-dn" % sites[0])
            return (rhoup - rhodn) / (2 * self._delta)
        if len(sites) == 2:
            i, j = sites
            rho = self.get_density_derivative([])
            rhoiup = self._cached_reader("QM/order-1/site-%d-up" % i)
            rhoidn = self._cached_reader("QM/order-1/site-%d-dn" % i)
            if i != j:
                rhojup = self._cached_reader("QM/order-1/site-%d-up" % j)
                rhojdn = self._cached_reader("QM/order-1/site-%d-dn" % j)
                rhoup = self._cached_reader(
                    "QM/order-2/site-%d-%d-up" % (min(i, j), max(i, j))
                )
                rhodn = self._cached_reader(
                    "QM/order-2/site-%d-%d-dn" % (min(i, j), max(i, j))
                )

            if i == j:
                deriv = (rhoiup + rhoidn - 2 * rho) / (self._delta ** 2)
            else:
                deriv = (
                    rhoup + rhodn + 2 * rho - rhoiup - rhoidn - rhojup - rhojdn
                ) / (2 * self._delta ** 2)
            return deriv
        raise NotImplementedError()

    def get_density_from_reference(self, nuclear_charges):
        return self._cached_reader(
            "QM/comparison-%s" % "-".join(map(str, nuclear_charges))
        )

    def get_energy_from_reference(self, nuclear_charges, is_reference_molecule=False):
        if is_reference_molecule:
            return self._calculator.get_total_energy("QM/order-0/site-all-cc")
        else:
            return self._calculator.get_total_energy(
                "QM/comparison-%s" % ("-".join(map(str, nuclear_charges)))
            )

    def analyse(self, explicit_reference=False, do_energies=True, do_dipoles=True):
        """ Performs actual analysis and integration. Prints results"""
        try:
            targets, energies, dipoles = self.predict_all_targets(
                do_energies, do_dipoles
            )
        except (FileNotFoundError, AttributeError):
            apdft.log.log(
                "At least one of the QM calculations has not been performed yet. Please run all QM calculations first.",
                level="warning",
            )
            return

        if explicit_reference:
            comparison_energies = np.zeros(len(targets))
            comparison_dipoles = np.zeros((len(targets), 3))
            for targetidx, target in enumerate(targets):
                path = "QM/comparison-%s" % "-".join(map(str, target))
                comparison_energies[targetidx] = self._calculator.get_total_energy(path)

                rho = self._cached_reader(path)
                nd = apdft.physics.Dipoles.point_charges(
                    [0, 0, 0], self._coordinates, target
                )
                ed = apdft.physics.Dipoles.electron_density(
                    [0, 0, 0], self._gridcoords, rho * self._gridweights
                )
                comparison_dipoles[targetidx] = ed + nd
        else:
            comparison_energies = None
            comparison_dipoles = None

        self._print_energies(targets, energies, comparison_energies)
        self._print_dipoles(targets, dipoles, comparison_dipoles)

        # persist results to disk
        targetnames = [DerivativeFolders._get_target_name(_) for _ in targets]
        result_energies = {"targets": targetnames, "total_energy": energies}
        result_dipoles = {
            "targets": targetnames,
            "dipole_moment_x": dipoles[:, 0],
            "dipole_moment_y": dipoles[:, 1],
            "dipole_moment_z": dipoles[:, 2],
        }
        if explicit_reference:
            result_energies["reference_energy"] = comparison_energies
            result_dipoles["reference_dipole_x"] = comparison_dipoles[:, 0]
            result_dipoles["reference_dipole_y"] = comparison_dipoles[:, 1]
            result_dipoles["reference_dipole_z"] = comparison_dipoles[:, 2]
        pd.DataFrame(result_energies).to_csv("energies.csv")
        pd.DataFrame(result_dipoles).to_csv("dipoles.csv")

        return targets, energies, comparison_energies
