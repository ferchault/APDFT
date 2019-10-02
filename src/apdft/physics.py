#!/usr/bin/env python
import numpy as np
import basis_set_exchange as bse
import pyscf
from pyscf import dft
import apdft
import os
import itertools as it
import pandas as pd

#: Conversion factor from Bohr in Angstrom.
angstrom = 1 / 0.52917721067
#: Conversion factor from electron charges and Angstrom to Debye
debye = 1 / 0.20819433


class Coulomb(object):
    """ Collects functions for Coulomb interaction."""

    @staticmethod
    def nuclei_nuclei(coordinates, charges):
        """ Calculates the nuclear-nuclear interaction energy from Coulomb interaction.

		Sign convention assumes positive charges for nuclei.

		Args:
			coordinates:		A (3,N) array of nuclear coordinates :math:`\\mathbf{r_i}`. [Angstrom]
			charges:			A N array of point charges :math:`q_i`. [e]
		Returns:
			Coulombic interaction energy. [Hartree]
		"""
        natoms = len(coordinates)
        ret = 0.0
        for i in range(natoms):
            for j in range(i + 1, natoms):
                d = np.linalg.norm((coordinates[i] - coordinates[j]) * angstrom)
                ret += charges[i] * charges[j] / d
        return ret

    @staticmethod
    def nuclear_potential(coordinates, charges, at):
        natoms = len(coordinates)
        ret = 0.0
        for i in range(natoms):
            if i == at:
                continue
            d = np.linalg.norm((coordinates[i] - coordinates[at]) * angstrom)
            ret += charges[i] / d
        return ret


class Dipoles(object):
    """ Collects functions regarding the calculation of dipole moments. This code follows the physics convention of the sign: the dipole moment vector points from the negative charge center to the positive charge center."""

    @staticmethod
    def point_charges(reference_point, coordinates, charges):
        """ Calculates the dipole moment of point charges.

		Note that for sets of point charges of a net charge, the resulting dipole moment depends on the chosen reference point. A common choice in the molecular context is the center of mass.
		Sign convention is such that nuclei should be given as positive charges.

		.. math::

			\\mathbf{p}(\\mathbf{r}) = \\sum_I q_i(\\mathbf{r_i}-\\mathbf{r})
		
		Args:
			reference_point:	A 3 array of the reference point :math:`\\mathbf{r}`. [Angstrom]
			coordinates: 		A (3,N) array of point charge coordinates :math:`\\mathbf{r_i}`. [Angstrom]
			charges:			A N array of point charges :math:`q_i`. [e]
		Returns:
			Dipole moment :math:`\\mathbf{p}`. [Debye]
		"""
        shift = coordinates - reference_point
        return np.sum(shift.T * charges, axis=1) * debye

    @staticmethod
    def electron_density(reference_point, coordinates, electron_density):
        """ Calculates the dipole moment of a charge distribution.

		Note that for a charge density, the resulting dipole momennt depends on the chosen reference point. A common choice in the molecular context is the center of mass.
		Electron density is a positive quantity.

		.. math::

			\\mathbf{p}(\\mathbf{r}) = \\int_\\Omega \\rho(\\mathbf{r_i})(\\mathbf{r_i}-\\mathbf{r})
		
		Args:
			reference_point:	A 3 array of the reference point :math:`\\mathbf{r}`. [Angstrom]
			coordinates: 		A (3,N) array of grid coordinates :math:`\\mathbf{r_i}`. [Angstrom]
			electron_density:	A N array of electron density values :math:`\\rho` at `coordinates`. [e/Angstrom^3]
		Returns:
			Dipole moment :math:`\\mathbf{p}`. [Debye]
		"""
        shift = coordinates - reference_point
        return -np.sum(shift.T * electron_density, axis=1) * debye


def charge_to_label(Z):
    """ Converts a nuclear charge to an element label.

	Uncharged (ghost) sites are assigned a dash.

	Args:
		Z 					Nuclear charge. [e]
	Returns:
		Element label. [String]
	"""
    if Z == 0:
        return "-"
    return bse.lut.element_sym_from_Z(Z, normalize=True)


class APDFT(object):
    """ Implementation of alchemical perturbation density functional theory.
    
    Requires a working directory `basepath` which allows for storing the intermediate calculation results."""

    def __init__(
        self,
        highest_order,
        nuclear_numbers,
        coordinates,
        basepath,
        calculator,
        max_charge=0,
        max_deltaz=3,
        include_atoms=None,
        targetlist=None,
    ):
        if highest_order > 2:
            raise NotImplementedError()
        self._orders = list(range(0, highest_order + 1))
        self._nuclear_numbers = np.array(nuclear_numbers)
        self._coordinates = coordinates
        self._delta = 0.05
        self._basepath = basepath
        self._calculator = calculator
        self._max_charge = max_charge
        self._max_deltaz = max_deltaz
        self._targetlist = targetlist
        if include_atoms is None:
            self._include_atoms = list(range(len(self._nuclear_numbers)))
        else:
            included = []
            for part in include_atoms:
                if type(part) == int:
                    included.append(part)
                else:
                    included += list(
                        np.where(
                            self._nuclear_numbers == bse.lut.element_Z_from_sym(part)
                        )[0]
                    )
            self._include_atoms = sorted(list(set(included)))

    def _calculate_delta_Z_vector(self, numatoms, order, sites, direction):
        baseline = np.zeros(numatoms)

        if order > 0:
            sign = {"up": 1, "dn": -1}[direction] * self._delta
            baseline[list(sites)] += sign

        return baseline

    def prepare(self, explicit_reference=False):
        """ Builds a complete folder list of all relevant calculations."""
        if os.path.isdir("QM"):
            apdft.log.log(
                "Input folder exists. Reusing existing data.", level="warning"
            )

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
                    commands.append("( cd %s && bash run.sh )" % path)
                    if os.path.isdir(path):
                        continue
                    os.makedirs(path)

                    charges = self._nuclear_numbers + self._calculate_delta_Z_vector(
                        len(self._nuclear_numbers), order, combination, direction
                    )
                    inputfile = self._calculator.get_input(
                        self._coordinates,
                        self._nuclear_numbers,
                        charges,
                        None,
                        includeonly=self._include_atoms,
                    )
                    with open("%s/run.inp" % path, "w") as fh:
                        fh.write(inputfile)
                    with open("%s/run.sh" % path, "w") as fh:
                        fh.write(
                            self._calculator.get_runfile(
                                self._coordinates, self._nuclear_numbers, charges, None
                            )
                        )

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

    def _get_stencil_coefficients(self, deltaZ, shift):
        """ Calculates the prefactors of the density terms outlined in the documentation of the implementation, e.g. alpha and beta.

        In general, this collects all terms of the taylor expansion for one particular target and returns their coefficients.
        For energies, the n-th order derivative of the density is divided by :math:`(n+1)!`, while the target density is obtained
        from a regular Taylor expansion, i.e. the density derivative is divided by :math:`n!`. Therefore, a `shift` of 1 returns
        energy coefficients and a `shift` of 0 returns density coefficients.

        Args:
            self:   APDFT instance to obtain the stencil from.
            deltaZ: Array of length N. Target system as described by the change in nuclear charges. [e]
            shift:  Integer. Shift of the factorial term in the energy expansion or density expansion.
        """

        # build alphas
        N = len(self._include_atoms)
        nvals = {0: 1, 1: N * 2, 2: N * (N - 1)}
        alphas = np.zeros(sum([nvals[_] for _ in self._orders]))

        # order 0
        if 0 in self._orders:
            alphas[0] = 1

        # order 1
        if 1 in self._orders:
            prefactor = 1 / (2 * self._delta) / np.math.factorial(1 + shift)
            for siteidx in range(N):
                alphas[1 + siteidx * 2] += prefactor * deltaZ[siteidx]
                alphas[1 + siteidx * 2 + 1] -= prefactor * deltaZ[siteidx]

        # order 2
        if 2 in self._orders:
            pos = 1 + N * 2 - 2
            for siteidx_i in range(N):
                for siteidx_j in range(siteidx_i, N):
                    if siteidx_i != siteidx_j:
                        pos += 2
                    if deltaZ[siteidx_j] == 0 or deltaZ[siteidx_i] == 0:
                        continue
                    if self._include_atoms[siteidx_j] > self._include_atoms[siteidx_i]:
                        prefactor = (1 / (2 * self._delta ** 2)) / np.math.factorial(
                            2 + shift
                        )
                        prefactor *= deltaZ[siteidx_i] * deltaZ[siteidx_j]
                        alphas[pos] += prefactor
                        alphas[pos + 1] += prefactor
                        alphas[0] += 2 * prefactor
                        alphas[1 + siteidx_i * 2] -= prefactor
                        alphas[1 + siteidx_i * 2 + 1] -= prefactor
                        alphas[1 + siteidx_j * 2] -= prefactor
                        alphas[1 + siteidx_j * 2 + 1] -= prefactor
                    if self._include_atoms[siteidx_j] == self._include_atoms[siteidx_i]:
                        prefactor = (1 / (self._delta ** 2)) / np.math.factorial(
                            2 + shift
                        )
                        prefactor *= deltaZ[siteidx_i] * deltaZ[siteidx_j]
                        alphas[0] -= 2 * prefactor
                        alphas[1 + siteidx_i * 2] += prefactor
                        alphas[1 + siteidx_j * 2 + 1] += prefactor

        return alphas

    def get_epn_coefficients(self, deltaZ):
        """ EPN coefficients are the weighting of the electronic EPN from each of the finite difference calculations.
        
        The weights depend on the change in nuclear charge, i.e. implicitly on reference and target molecule as well as the finite difference stencil employed."""
        return self._get_stencil_coefficients(deltaZ, 1)

    def _print_energies(self, targets, energies, comparison_energies):
        if comparison_energies is None:
            for target, energy in zip(targets, energies):
                targetname = APDFT._get_target_name(target)
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
                targetname = APDFT._get_target_name(target)
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
        if comparison_dipoles is None:
            for target, dipole in zip(targets, dipoles):
                targetname = APDFT._get_target_name(target)
                apdft.log.log(
                    "Dipole calculated",
                    level="RESULT",
                    kind="total_dipole",
                    value=list(dipole),
                    target=target,
                    targetname=targetname,
                )
        else:
            for target, dipole, comparison in zip(targets, dipoles, comparison_dipoles):
                targetname = APDFT._get_target_name(target)
                apdft.log.log(
                    "Dipole calculated",
                    level="RESULT",
                    kind="total_dipole",
                    reference=list(comparison),
                    value=list(dipole),
                    target=target,
                    targetname=targetname,
                )

    @staticmethod
    def _get_target_name(target):
        return ",".join([apdft.physics.charge_to_label(_) for _ in target])

    def get_folder_order(self):
        """ Returns a static order of calculation folders to build the individual derivative entries.

        To allow for a more efficient evaluation of APDFT, terms are collected and most of the evaluation
        is done with the combined cofficients of those terms. This requires the terms to be handled in a certain
        fixed order that is stable in the various parts of the code. Depending on the selected expansion order, this 
        function builds the list of folders to be included.

        Returns: List of strings, the folder names."""

        folders = []

        # order 0
        folders.append("%s/QM/order-0/site-all-cc/" % self._basepath)

        # order 1
        if 1 in self._orders:
            for site in self._include_atoms:
                folders.append("%s/QM/order-1/site-%d-up/" % (self._basepath, site))
                folders.append("%s/QM/order-1/site-%d-dn/" % (self._basepath, site))

        # order 2
        if 2 in self._orders:
            for site_i in self._include_atoms:
                for site_j in self._include_atoms:
                    if site_j <= site_i:
                        continue

                    folders.append(
                        "%s/QM/order-2/site-%d-%d-up/"
                        % (self._basepath, site_i, site_j)
                    )
                    folders.append(
                        "%s/QM/order-2/site-%d-%d-dn/"
                        % (self._basepath, site_i, site_j)
                    )

        return folders

    def get_epn_matrix(self):
        """ Collects :math:`\int_Omega rho_i(\mathbf{r}) /|\mathbf{r}-\mathbf{R}_I|`. """
        N = len(self._include_atoms)
        folders = self.get_folder_order()

        coeff = np.zeros((len(folders), N))

        def get_epn(folder, order, direction, combination):
            res = 0.0
            charges = self._nuclear_numbers + self._calculate_delta_Z_vector(
                len(self._nuclear_numbers), order, combination, direction
            )
            try:
                res = self._calculator.get_epn(
                    folder, self._coordinates, self._include_atoms, charges
                )
            except ValueError:
                apdft.log.log(
                    "Calculation with incomplete results.",
                    level="error",
                    calulation=folder,
                )
            except FileNotFoundError:
                apdft.log.log(
                    "Calculation is missing a result file.",
                    level="error",
                    calculation=folder,
                )
            return res

        # order 0
        pos = 0

        # order 0
        coeff[pos, :] = get_epn(folders[pos], 0, "up", 0)
        pos += 1

        # order 1
        if 1 in self._orders:
            for site in self._include_atoms:
                coeff[pos, :] = get_epn(folders[pos], 1, "up", [site])
                coeff[pos + 1, :] = get_epn(folders[pos + 1], 1, "dn", [site])
                pos += 2

        # order 2
        if 2 in self._orders:
            for site_i in self._include_atoms:
                for site_j in self._include_atoms:
                    if site_j <= site_i:
                        continue

                    coeff[pos, :] = get_epn(folders[pos], 2, "up", [site_i, site_j])
                    coeff[pos + 1, :] = get_epn(
                        folders[pos + 1], 2, "dn", [site_i, site_j]
                    )
                    pos += 2

        return coeff

    def get_linear_density_coefficients(self, deltaZ):
        """ Obtains the finite difference coefficients for a property linear in the density. 
        
        Args:
            deltaZ:     Array of integers of length N. Target system expressed in the change in nuclear charges from the reference system. [e]
        Returns:
            Vector of coefficients."""
        return self._get_stencil_coefficients(deltaZ, 0)

    def enumerate_all_targets(self):
        """ Builds a list of all possible targets.

		Note that the order is not guaranteed to be stable.

		Args:
			self:		Class instance from which the total charge and number of sites is determined.
		Returns:
			A list of lists with the integer nuclear charges."""
        # there might be a user-specified explicit list
        if self._targetlist is not None:
            return self._targetlist

        # Generate targets
        if self._max_deltaz is None:
            around = None
            limit = None
        else:
            around = np.array(self._nuclear_numbers)
            limit = self._max_deltaz

        res = []
        nsites = len(self._nuclear_numbers)
        nprotons = sum(self._nuclear_numbers)
        for shift in range(-self._max_charge, self._max_charge + 1):
            if nprotons + shift < 1:
                continue
            res += apdft.math.IntegerPartitions.partition(
                nprotons + shift, nsites, around, limit
            )

        # filter for included atoms
        ignore_atoms = list(
            set(range(len(self._nuclear_numbers))) - set(self._include_atoms)
        )
        if len(self._include_atoms) != len(self._nuclear_numbers):
            res = [
                _
                for _ in res
                if [_[idx] for idx in ignore_atoms]
                == [self._nuclear_numbers[idx] for idx in ignore_atoms]
            ]
        return res

    def estimate_cost_and_coverage(self):
        """ Estimates number of single points (cost) and number of targets (coverage).

		Args:
			self:		Class instance from which the total charge and number of sites is determined.
		Returns:
			Tuple of ints: number of single points, number of targets."""

        N = len(self._include_atoms)
        cost = sum({0: 1, 1: 2 * N, 2: N * (N - 1)}[_] for _ in self._orders)

        coverage = len(self.enumerate_all_targets())
        return cost, coverage

    def get_energy_from_reference(self, nuclear_charges, is_reference_molecule=False):
        """ Retreives the total energy from a QM reference. 

		Args:
			nuclear_charges: 	Integer list of nuclear charges. [e]
		Returns:
			The total energy. [Hartree]"""
        if is_reference_molecule:
            return self._calculator.get_total_energy("QM/order-0/site-all-cc")
        else:
            return self._calculator.get_total_energy(
                "QM/comparison-%s" % ("-".join(map(str, nuclear_charges)))
            )

    def get_linear_density_matrix(self, propertyname):
        """ Retrieves the value matrix for properties linear in density.

        Valid properties are: ELECTRONIC_DIPOLE, IONIC_FORCE, ELECTRONIC_QUADRUPOLE.
        Args:
            self:           APDFT instance.
            propertyname:   String. One of the choices above.
        Returns: 
            (N, m) array for an m-dimensional property over N QM calculations or None if the property is not implemented with this QM code."""

        functionname = "get_%s" % propertyname.lower()
        try:
            function = getattr(self._calculator, functionname)
        except AttributeError:
            return None

        folders = self.get_folder_order()
        results = [function(folder) for folder in folders]
        return np.array(results)

    def predict_all_targets(self):
        # assert one order of targets
        targets = self.enumerate_all_targets()
        own_nuc_nuc = Coulomb.nuclei_nuclei(self._coordinates, self._nuclear_numbers)

        energies = np.zeros(len(targets))
        dipoles = np.zeros((len(targets), 3))
        natoms = len(self._coordinates)

        # get base information
        refenergy = self.get_energy_from_reference(
            self._nuclear_numbers, is_reference_molecule=True
        )
        epn_matrix = self.get_epn_matrix()
        dipole_matrix = self.get_linear_density_matrix("ELECTRONIC_DIPOLE")

        # get target predictions
        for targetidx, target in enumerate(targets):
            deltaZ = target - self._nuclear_numbers

            alphas = self.get_epn_coefficients(deltaZ)
            deltaZ_included = deltaZ[self._include_atoms]
            deltaE = -np.sum(np.multiply(np.outer(alphas, deltaZ_included), epn_matrix))
            deltaE += Coulomb.nuclei_nuclei(self._coordinates, target) - own_nuc_nuc
            energies[targetidx] = deltaE + refenergy

            if dipole_matrix is not None:
                betas = self.get_linear_density_coefficients(deltaZ)
                nuc_dipole = Dipoles.point_charges(
                    self._coordinates.mean(axis=0), self._coordinates, target
                )
                ed = np.multiply(dipole_matrix, betas[:, np.newaxis]).sum(axis=0)
                dipoles[targetidx] = ed + nuc_dipole

        # return results
        return targets, energies, dipoles

    def analyse(self, explicit_reference=False):
        """ Performs actual analysis and integration. Prints results"""
        try:
            targets, energies, dipoles = self.predict_all_targets()
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
                try:
                    comparison_energies[targetidx] = self._calculator.get_total_energy(
                        path
                    )
                except FileNotFoundError:
                    apdft.log.log(
                        "Comparison calculation is missing. Predictions are unaffected. Will skip this comparison.",
                        level="warning",
                        calculation=path,
                        target=target,
                    )
                    comparison_energies[targetidx] = np.nan
                    comparison_dipoles[targetidx] = np.nan
                    continue

                nd = apdft.physics.Dipoles.point_charges(
                    [0, 0, 0], self._coordinates, target
                )
                # TODO: load dipole
                # comparison_dipoles[targetidx] = ed + nd
        else:
            comparison_energies = None
            comparison_dipoles = None

        self._print_energies(targets, energies, comparison_energies)
        self._print_dipoles(targets, dipoles, comparison_dipoles)

        # persist results to disk
        targetnames = [APDFT._get_target_name(_) for _ in targets]
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
        pd.DataFrame(result_energies).to_csv("energies.csv", index=False)
        pd.DataFrame(result_dipoles).to_csv("dipoles.csv", index=False)

        return targets, energies, comparison_energies
