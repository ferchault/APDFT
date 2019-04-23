#!/usr/bin/env python

class Dipoles(object):
	""" Collects functions regarding the calculation of dipole moments."""

	debye = 0.20819433

	@staticmethod
	def point_charges(reference_point, coordinates, charges):
		""" Calculates the dipole moment of point charges.

		.. math::

			\mathbf{p}(r) = \sum_I q_i(r-r_i)
		
		Args:
			reference_point:	A 3 array of the reference point. [Angstrom]
			coordinates: 		A (3,N) array of point charge coordinates. [Angstrom]
			charges:			A N array of point charges. [e]
		Returns:
			Dipole moment. [Debye]
		"""
		pass

