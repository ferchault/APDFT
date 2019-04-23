#!/usr/bin/env python

class Dipoles(object):
	""" Collects functions regarding the calculation of dipole moments."""

	debye = 0.20819433

	@staticmethod
	def point_charges(reference_point, coordinates, charges):
		""" Calculates the dipole moment of point charges.

		.. math::

			\mathbf{p}(\mathbf{r}) = \sum_I q_i(\mathbf{r_i}-\mathbf{r})
		
		Args:
			reference_point:	A 3 array of the reference point :math:`\mathbf{r}`. [Angstrom]
			coordinates: 		A (3,N) array of point charge coordinates :math:`\mathbf{r_i}`. [Angstrom]
			charges:			A N array of point charges :math:`q_i`. [e]
		Returns:
			Dipole moment :math:`\mathbf{p}`. [Debye]
		"""
		pass

