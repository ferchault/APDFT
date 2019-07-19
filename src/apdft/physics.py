#!/usr/bin/env python
import numpy as np
import basis_set_exchange as bse
import pyscf
from pyscf import dft
import apdft

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
                ret = charges[i] * charges[j] / d
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

	This is an abstract base class. This means that for any use, one needs to inherit from this class.
	The subclass needs to implement all functions that raise a NotImplementedError upon invocation."""

    def __init__(
        self,
        highest_order,
        nuclear_numbers,
        coordinates,
        max_charge=0,
        max_deltaz=3,
        include_atoms=None,
    ):
        if highest_order > 2:
            raise NotImplementedError()
        self._orders = list(range(0, highest_order + 1))
        self._nuclear_numbers = nuclear_numbers
        self._coordinates = coordinates
        self._reader_cache = dict()
        self._delta = 0.05
        self._max_charge = max_charge
        self._max_deltaz = max_deltaz
        self._gridweights = None
        self._gridcoords = None
        if include_atoms is None:
            self._include_atoms = list(range(len(self._nuclear_numbers)))
        else:
            self._include_atoms = include_atoms

    def update_grid(self):
        """ Ensures that the current integration grid is initialised."""
        raise NotImplementedError()

    def enumerate_all_targets(self):
        """ Builds a list of all possible targets.

		Note that the order is not guaranteed to be stable.

		Args:
			self:		Class instance from which the total charge and number of sites is determined.
		Returns:
			A list of lists with the integer nuclear charges."""
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
        if len(self._include_atoms) != len(self._nuclear_numbers):
            res = [
                _
                for _ in res
                if [_[idx] for idx in self._include_atoms]
                == [self._nuclear_numbers[idx] for idx in self._include_atoms]
            ]
        return res

    def estimate_cost_and_coverage(self):
        """ Estimates number of single points (cost) and number of targets (coverage).

		Args:
			self:		Class instance from which the total charge and number of sites is determined.
		Returns:
			Tuple of ints: number of single points, number of targets."""

        N = len(self._include_atoms)
        cost = sum({0: 1, 1: N, 2: 2 * N * N - 2 * N}[_] for _ in self._orders)

        coverage = len(self.enumerate_all_targets())
        return cost, coverage

    def get_energy_from_reference(self, nuclear_charges):
        """ Retreives the total energy from a QM reference. Abstract function.

		Light function, will not be called often, so no caching needed.

		Args:
			nuclear_charges: 	Integer list of nuclear charges. [e]
		Returns:
			The total energy. [Hartree]"""
        raise NotImplementedError()

    def get_density_from_reference(self, nuclear_charges):
        """ Retreives the density from a QM reference. Abstract function.

		Light function, will not be called often, so no caching needed.

		Args:
			nuclear_charges: 	Integer list of nuclear charges. [e]
			gridcoords: 		Grid coordinates. [Angstrom]
		Returns:
			A numpy array of electron density at the grid coordinates."""
        raise NotImplementedError()

    def get_density_derivative(self, sites):
        """ Retrieves the n-th order density derivative.

		Heavy function, will be called often, caching needed. 
		The order of the derivative is implicitly known since it's the length of *sites* argument via the chain rule.

		Args:
			sites:				Integer list of sites that are perturbed.
		Returns:
			A numpy array of electron density at the grid coordinates."""
        raise NotImplementedError()

    def predict_all_targets(self, do_energies=True, do_dipoles=True):
        # assert one order of targets
        targets = self.enumerate_all_targets()
        self.update_grid()
        own_nuc_nuc = Coulomb.nuclei_nuclei(self._coordinates, self._nuclear_numbers)

        # allocate output
        if do_energies:
            energies = np.zeros(len(targets))
        else:
            energies = None
        if do_dipoles:
            dipoles = np.zeros((len(targets), 3))
        else:
            dipoles = None
        natoms = len(self._coordinates)

        # get base information
        grid_ds = np.linalg.norm(self._gridcoords * apdft.physics.angstrom, axis=1)
        ds = []
        for site in self._coordinates:
            ds.append(
                np.linalg.norm(
                    (self._gridcoords - site) * apdft.physics.angstrom, axis=1
                )
            )
        refenergy = self.get_energy_from_reference(
            self._nuclear_numbers, is_reference_molecule=True
        )

        # get target predictions
        for targetidx, target in enumerate(targets):
            deltaZ = target - self._nuclear_numbers

            deltaV = np.zeros(len(self._gridweights))
            for atomidx in range(natoms):
                deltaV += deltaZ[atomidx] / ds[atomidx]

            # zeroth order
            rho = self.get_density_derivative([])
            rhotilde = rho.copy()
            rhotarget = rho.copy()

            # first order
            for atomidx in range(natoms):
                deriv = self.get_density_derivative([atomidx])
                rhotilde += deriv * deltaZ[atomidx] / 2
                rhotarget += deriv * deltaZ[atomidx]

            # second order
            for i in range(natoms):
                for j in range(natoms):
                    deriv = self.get_density_derivative([i, j])
                    rhotilde += (deriv * deltaZ[i] * deltaZ[j]) / 6
                    rhotarget += (deriv * deltaZ[i] * deltaZ[j]) / 2

            d_nuc_nuc = Coulomb.nuclei_nuclei(self._coordinates, target) - own_nuc_nuc
            energies[targetidx] = (
                -np.sum(rhotilde * deltaV * self._gridweights) + d_nuc_nuc + refenergy
            )
            nuc_dipole = Dipoles.point_charges([0, 0, 0], self._coordinates, target)
            ed = Dipoles.electron_density(
                [0, 0, 0], self._gridcoords, rhotarget * self._gridweights
            )
            dipoles[targetidx] = ed + nuc_dipole

        # return results
        return targets, energies, dipoles
