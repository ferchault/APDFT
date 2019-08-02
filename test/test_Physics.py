#!/usr/bin/env python
import pytest
import numpy as np

import apdft.physics as phys

def test_dipole_magnitude():
	coordinates = np.array([[0., 0., 1.], [0., 0., 0.]])
	charges = np.array([1, -1])
	dipole = phys.Dipoles.point_charges(np.array([0., 0., 0.]), coordinates, charges)
	assert abs(np.linalg.norm(dipole) - 1/0.20819433) < 1e-7

def test_dipole_sign():
	coordinates = np.array([[0., 0., 1.], [0., 0., 0.]])
	charges = np.array([1, -1])
	dipole = phys.Dipoles.point_charges(np.array([0., 0., 0.]), coordinates, charges)
	assert dipole[-1] > 0

	charges = np.array([1, -1])
	dipole = phys.Dipoles.electron_density(np.array([0., 0., 0.]), coordinates, charges)
	assert dipole[-1] < 0

def test_coulomb_nuclear_nuclear():
	coordinates = np.array([[0., 0., 0.52917721067], [0., 0., 0.]])
	charges = np.array([1, 1])
	assert phys.Coulomb.nuclei_nuclei(coordinates, charges) == 1

	coordinates = np.array([[0., 0., 0.52917721067], [0., 0., 0.], [0., 0., -0.52917721067]])
	charges = np.array([1, 1, 1])
	assert phys.Coulomb.nuclei_nuclei(coordinates, charges) == 2.5

def test_coulomb_nuclear_nuclear_sign():
	coordinates = np.array([[0., 0., 0.52917721067], [0., 0., 0.]])
	charges = np.array([1, -1])
	assert phys.Coulomb.nuclei_nuclei(coordinates, charges) == -1

def test_element_conversion():
	assert phys.charge_to_label(0) == "-"
	assert phys.charge_to_label(1) == "H"
	assert phys.charge_to_label(2) == "He"

def test_include_list_element():
	a = phys.APDFT(2, [1, 1, 1, 6, 6, 6], np.zeros((3, 6)), include_atoms=[0, 'C'])
	assert (a._include_atoms == [0, 3, 4, 5])

def test_include_list_duplicate():
	a = phys.APDFT(2, [1, 1, 1, 6, 6, 6], np.zeros((3, 6)), include_atoms=[4, 'C'])
	assert (a._include_atoms == [3, 4, 5])
