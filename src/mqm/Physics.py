#!/usr/bin/env python
import numpy as np

class Dipoles(object):
	""" Collects functions regarding the calculation of dipole moments."""

	debye = 0.20819433

	@staticmethod
	def point_charges(reference_point, coordinates, charges):
		""" Calculates the dipole moment of point charges.

		Note that for sets of point charges of a net charge, the resulting dipole moment depends on the chosen reference point. A common choice in the molecular context is the center of mass.

		.. math::

			\mathbf{p}(\mathbf{r}) = \sum_I q_i(\mathbf{r_i}-\mathbf{r})
		
		Args:
			reference_point:	A 3 array of the reference point :math:`\mathbf{r}`. [Angstrom]
			coordinates: 		A (3,N) array of point charge coordinates :math:`\mathbf{r_i}`. [Angstrom]
			charges:			A N array of point charges :math:`q_i`. [e]
		Returns:
			Dipole moment :math:`\mathbf{p}`. [Debye]
		"""
		shift = coordinates - reference_point
		return np.sum(shift.T * charges, axis=0) * Dipoles.debye
