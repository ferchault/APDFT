#!/usr/bin/env python
import numpy as np
import basis_set_exchange as bse
import pyscf
from pyscf import dft
import mqm

#: Conversion factor from Bohr in Angstrom.
angstrom = 1/0.52917721067
#: Conversion factor from electron charges and Angstrom to Debye
debye = 1/0.20819433

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
		ret = 0.
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
		return '-'
	return bse.lut.element_sym_from_Z(Z, normalize=True)

class APDFT(object):
	""" Implementation of alchemical perturbation density functional theory."""
	def __init__(self, highest_order, nuclear_numbers, coordinates):
		if highest_order > 2:
			raise NotImplementedError()
		self._orders = list(range(0, highest_order+1))
		self._nuclear_numbers = nuclear_numbers
		self._coordinates = coordinates
		self._reader_cache = dict()
		self._delta = 0.05

	def _get_grid(self):
		""" Returns the integration grid in Angstrom."""
		mol = pyscf.gto.Mole()
		for nuclear, coord in zip(self._nuclear_numbers, self._coordinates):
			# pyscf molecule init is in Angstrom
			mol.atom.extend([[nuclear, *coord]])
		mol.build()
		grid = dft.gen_grid.Grids(mol)
		grid.level = 3
		grid.build()
		# pyscf grid is in a.u.
		return grid.coords/angstrom, grid.weights

	def enumerate_all_targets(self, max_charge=2):
		""" Builds a list of all possible targets.

		Note that the order is not guaranteed to be stable.

		Args:
			self:		Class instance from which the total charge and numebr of sites is determined.
			max_charge:	Maxmimum absolute molecular charge allowed. [e]
		Returns:
			A list of lists with the integer nuclear charges."""
		res = []
		nsites = len(self._nuclear_numbers)
		nprotons = sum(self._nuclear_numbers)
		for shift in range(-max_charge, max_charge + 1):
			if nprotons + shift < 1:
				continue
			res += mqm.math.IntegerPartitions.partition(nprotons + shift, nsites)
		return res
