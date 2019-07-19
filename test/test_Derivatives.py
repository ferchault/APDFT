#!/usr/bin/env python
import pytest
import glob
import os
import shutil

import numpy as np

import apdft
import apdft.Derivatives as apd
import apdft.calculator as apc
from apdft.calculator.gaussian import GaussianCalculator

@pytest.fixture
def mock_derivatives():
	d = apd.DerivativeFolders(0, [2, 2], np.array([[0, 0, 1], [0, 0, 2]]), 0, 5)
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
	derivatives = apdft.Derivatives.DerivativeFolders(2, nuclear_numbers, coordinates, 0, 50)
	derivatives.assign_calculator(calculator)
	targets, energies, comparison_energies = derivatives.analyse(explicit_reference=True)

	# check one energy value
	lookup  = [1, 13]
	pos = targets.index(lookup)
	assert abs(energies[pos] - -160.15390113953077) < 1e-7
	assert abs(comparison_energies[pos] - -177.78263968061) < 1e-7

	# without reference
	derivatives = apdft.Derivatives.DerivativeFolders(2, nuclear_numbers, coordinates, 0, 50)
	derivatives.assign_calculator(calculator)
	targets, energies, comparison_energies = derivatives.analyse(explicit_reference=False)

	# check one energy value
	lookup  = [1, 13]
	pos = targets.index(lookup)
	assert abs(energies[pos] - -160.15390113953077) < 1e-7
	assert comparison_energies is None

	os.chdir(pwd)

def test_grid():
	c = apc.MockCalculator('HF', 'STO-3G')
	d = apd.DerivativeFolders(0, [1, 1], [[0, 0, 1], [0, 0, 2]])
	d.assign_calculator(c)
	coords, weights = d._get_grid()
	center = np.average(coords, axis=0)
	assert center[0] - 0 < 1e-8
	assert center[1] - 0 < 1e-8
	assert center[2] - 1.5 < 1e-8

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
	d = apd.DerivativeFolders(2, [2, 3], np.array([[0, 0, 1], [0, 0, 2]]), 0, 6)
	d.assign_calculator(c)
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
		d = apd.DerivativeFolders(3, [2, 3], np.array([[0, 0, 1], [0, 0, 2]]))

def test_restricted_atom_set():
	d = apd.DerivativeFolders(0, [2, 2], np.array([[0, 0, 1], [0, 0, 2]]), 0, 5, [0,])
	targets = set([tuple(_) for _ in d.enumerate_all_targets()])
	expected = set([(2, 2)])
	assert targets == expected
