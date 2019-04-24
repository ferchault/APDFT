#!/usr/bin/env python
import pytest
import numpy as np

import mqm.physics as phys

def test_dipole_magnitude():
	coordinates = np.array([[0., 0., 1.], [0., 0., 0.]])
	charges = np.array([1, -1])
	dipole = phys.Dipoles.point_charges(np.array([0., 0., 0.]), coordinates, charges)
	assert abs(np.linalg.norm(dipole) - 0.20819433) < 1e-7

def test_dipole_sign():
	coordinates = np.array([[0., 0., 1.], [0., 0., 0.]])
	charges = np.array([1, -1])
	dipole = phys.Dipoles.point_charges(np.array([0., 0., 0.]), coordinates, charges)
	assert dipole[-1] > 0

	charges = np.array([1, -1])
	dipole = phys.Dipoles.electron_density(np.array([0., 0., 0.]), coordinates, charges)
	assert dipole[-1] < 0