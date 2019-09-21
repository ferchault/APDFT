#!/usr/bin/env python
import pytest
import numpy as np

import pytest
import glob
import os
import shutil

import apdft
import apdft.calculator as apc
import apdft.physics as ap
from apdft.calculator.gaussian import GaussianCalculator
from apdft.calculator import MockCalculator

def test_dipole_magnitude():
	coordinates = np.array([[0., 0., 1.], [0., 0., 0.]])
	charges = np.array([1, -1])
	dipole = ap.Dipoles.point_charges(np.array([0., 0., 0.]), coordinates, charges)
	assert abs(np.linalg.norm(dipole) - 1/0.20819433) < 1e-7

def test_dipole_sign():
	coordinates = np.array([[0., 0., 1.], [0., 0., 0.]])
	charges = np.array([1, -1])
	dipole = ap.Dipoles.point_charges(np.array([0., 0., 0.]), coordinates, charges)
	assert dipole[-1] > 0

	charges = np.array([1, -1])
	dipole = ap.Dipoles.electron_density(np.array([0., 0., 0.]), coordinates, charges)
	assert dipole[-1] < 0

def test_coulomb_nuclear_nuclear():
	coordinates = np.array([[0., 0., 0.52917721067], [0., 0., 0.]])
	charges = np.array([1, 1])
	assert ap.Coulomb.nuclei_nuclei(coordinates, charges) == 1

	coordinates = np.array([[0., 0., 0.52917721067], [0., 0., 0.], [0., 0., -0.52917721067]])
	charges = np.array([1, 1, 1])
	assert ap.Coulomb.nuclei_nuclei(coordinates, charges) == 2.5

def test_coulomb_nuclear_nuclear_sign():
	coordinates = np.array([[0., 0., 0.52917721067], [0., 0., 0.]])
	charges = np.array([1, -1])
	assert ap.Coulomb.nuclei_nuclei(coordinates, charges) == -1

def test_element_conversion():
	assert ap.charge_to_label(0) == "-"
	assert ap.charge_to_label(1) == "H"
	assert ap.charge_to_label(2) == "He"

def test_include_list_element():
	a = ap.APDFT(2, [1, 1, 1, 6, 6, 6], np.zeros((3, 6)), '.', MockCalculator('method', 'basis_set'), include_atoms=[0, 'C'])
	assert (a._include_atoms == [0, 3, 4, 5])

def test_include_list_duplicate():
	a = ap.APDFT(2, [1, 1, 1, 6, 6, 6], np.zeros((3, 6)), '.', MockCalculator('method', 'basis_set'), include_atoms=[4, 'C'])
	assert (a._include_atoms == [3, 4, 5])

@pytest.fixture
def mock_derivatives():
	d = ap.APDFT(0, [2, 2], np.array([[0, 0, 1], [0, 0, 2]]), '.', MockCalculator('method', 'basis_set'), 0, 5)
	return d

@pytest.fixture(scope="module")
def sample_rundir():
	tmpdir = os.path.abspath(apc.Calculator._get_tempname())
	path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
	try:
		shutil.copytree(path + '/data/apdft-run', '%s/QM' % tmpdir)
	except shutil.Error as e:
		# on windows subsystem for linux, copying mistankenly reports E_BUSY
		validerrors = [_ for _ in e if not e[2].startswith('[Errno 16] Device or resource busy')]
		if len(validerrors) > 0:
			raise
	yield tmpdir
	shutil.rmtree(tmpdir)

def test_readfile(sample_rundir):
	pwd = os.path.abspath(os.getcwd())
	os.chdir(sample_rundir)

	calculator = GaussianCalculator('HF', 'STO-3G')
	nuclear_numbers, coordinates = apdft.read_xyz('QM/n2.xyz')

	# with reference
	derivatives = ap.APDFT(2, nuclear_numbers, coordinates, '.', calculator, 0, 3)
	targets, energies, comparison_energies = derivatives.analyse(explicit_reference=True)

	# check one energy value
	lookup  = [8, 6]
	pos = targets.index(lookup)
	print (energies)
	assert abs(energies[pos] - -108.88437251951585) < 1e-7
	assert abs(comparison_energies[pos] - -111.1436117) < 1e-7

	# without reference
	derivatives = ap.APDFT(2, nuclear_numbers, coordinates,'.', calculator, 0, 3)
	targets, energies, comparison_energies = derivatives.analyse(explicit_reference=False)

	# check one energy value
	lookup  = [8, 6]
	pos = targets.index(lookup)
	assert abs(energies[pos] - -108.88437251951585) < 1e-7
	assert comparison_energies is None

	os.chdir(pwd)

def test_targets(mock_derivatives):
	targets = set([tuple(_) for _ in mock_derivatives.enumerate_all_targets()])
	expected = set([(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)])
	assert targets == expected

def test_filecontents():
	pwd = os.path.abspath(os.getcwd())
	tmpdir = os.path.abspath(apc.Calculator._get_tempname())
	os.mkdir(tmpdir)
	os.chdir(tmpdir)

	c = GaussianCalculator('HF', 'STO-3G')
	d = ap.APDFT(2, [2, 3], np.array([[0, 0, 1], [0, 0, 2]]), '.', c, 0, 6)
	assert d._orders == [0, 1, 2]
	d.prepare(explicit_reference=True)

	def _get_Zs_from_file(fn):
		with open(fn) as fh:
			lines = fh.readlines()[-2:]
		zs = [float(_.split()[-1]) for _ in lines]
		return zs

	delta = 0.05
	assert _get_Zs_from_file('QM/order-0/site-all-cc/run.inp') == [2., 3.]

	assert _get_Zs_from_file('QM/order-1/site-0-up/run.inp') == [2.+delta, 3.]
	assert _get_Zs_from_file('QM/order-1/site-0-dn/run.inp') == [2.-delta, 3.]
	assert _get_Zs_from_file('QM/order-1/site-1-up/run.inp') == [2., 3.+delta]
	assert _get_Zs_from_file('QM/order-1/site-1-dn/run.inp') == [2., 3.-delta]

	assert _get_Zs_from_file('QM/order-2/site-0-1-up/run.inp') == [2.+delta, 3.+delta]
	assert _get_Zs_from_file('QM/order-2/site-0-1-dn/run.inp') == [2.-delta, 3.-delta]

	assert _get_Zs_from_file('QM/comparison-0-5/run.inp') == [0., 5.]
	assert _get_Zs_from_file('QM/comparison-1-4/run.inp') == [1., 4.]
	assert _get_Zs_from_file('QM/comparison-2-3/run.inp') == [2., 3.]
	assert _get_Zs_from_file('QM/comparison-3-2/run.inp') == [3., 2.]
	assert _get_Zs_from_file('QM/comparison-4-1/run.inp') == [4., 1.]
	assert _get_Zs_from_file('QM/comparison-5-0/run.inp') == [5., 0.]

	assert set(map(os.path.basename, glob.glob('QM/*'))) == set('order-0 order-1 order-2 comparison-0-5 comparison-1-4 comparison-2-3 comparison-3-2 comparison-4-1 comparison-5-0'.split())
	assert set(map(os.path.basename, glob.glob('QM/order-0/*'))) == set('site-all-cc'.split())
	assert set(map(os.path.basename, glob.glob('QM/order-1/*'))) == set('site-0-up site-0-dn site-1-up site-1-dn'.split())
	assert set(map(os.path.basename, glob.glob('QM/order-2/*'))) == set('site-0-1-up site-0-1-dn'.split())

	os.chdir(pwd)
	shutil.rmtree(tmpdir)

def test_too_high_order():
	with pytest.raises(NotImplementedError):
		ap.APDFT(3, [2, 3], np.array([[0, 0, 1], [0, 0, 2]]), '.', MockCalculator('method', 'basis_set'))

def test_restricted_atom_set():
	d = ap.APDFT(0, [2, 2], np.array([[0, 0, 1], [0, 0, 2]]), '.', MockCalculator('method', 'basis_set'), 0, 5, [0,])
	targets = set([tuple(_) for _ in d.enumerate_all_targets()])
	expected = set([(2, 2)])
	assert targets == expected

def test_folder_respect_order_settings():
	d = ap.APDFT(0, [1, 2, 3], np.zeros((3, 3)), '.', MockCalculator('method', 'basis_set'))
	assert (len(d.get_folder_order()) == 1)
	d = ap.APDFT(1, [1, 2, 3], np.zeros((3, 3)), '.', MockCalculator('method', 'basis_set'))
	assert (len(d.get_folder_order()) == 1 + 3*2)
	d = ap.APDFT(2, [1, 2, 3], np.zeros((3, 3)), '.', MockCalculator('method', 'basis_set'))
	assert (len(d.get_folder_order()) == 1 + 3*2 + 3*2)

def test_target_enumeration_limited_atom_selection():
	d = ap.APDFT(0, [1, 2], np.zeros((2, 3)), '.', MockCalculator('method', 'basis_set'), 1, 1)
	expected = set([(0, 2), (1, 1), (1, 2), (1, 3), (2, 2)])
	actual = set([tuple(_) for _ in d.enumerate_all_targets()])
	assert (actual == expected)
	d = ap.APDFT(0, [1, 2], np.zeros((2, 3)), '.', MockCalculator('method', 'basis_set'), 1, 1, [0])
	expected = set([(0, 2), (1, 2), (2, 2)])
	actual = set([tuple(_) for _ in d.enumerate_all_targets()])
	assert (actual == expected)