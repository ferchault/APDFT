#!/usr/bin/env python
import pytest
import glob
import os
import shutil

import numpy as np

import mqm.Derivatives as mqmd
import mqm.Calculator as mqmc

@pytest.fixture
def mock_derivatives():
	c = mqmc.MockCalculator()
	d = mqmd.Derivatives(c, 0, [2, 2], np.array([[0, 0, 1], [0, 0, 2]]), 'HF', 'STO-3G')
	return d

def test_grid():
	c = mqmc.MockCalculator()
	d = mqmd.Derivatives(c, 0, [1, 1], [[0, 0, 1], [0, 0, 2]], 'HF', 'STO-3G')
	coords, weights = d._get_grid()
	center = np.average(coords, axis=0)
	assert center[0] - 0 < 1e-8
	assert center[1] - 0 < 1e-8
	assert center[2] - 1.5 < 1e-8

def test_targets(mock_derivatives):
	targets = set([tuple(_) for _ in mock_derivatives._enumerate_all_targets()])
	expected = set([(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)])
	assert targets == expected

def test_nucnuc(mock_derivatives):
	diff = mock_derivatives.calculate_delta_nuc_nuc(np.array([1, 3]))
	expected = -0.52917721067
	assert abs(diff - expected) < 1e-8

def test_filecontents():
	c = mqmc.GaussianCalculator()
	d = mqmd.DerivativeFolders(c, 2, [2, 3], np.array([[0, 0, 1], [0, 0, 2]]), 'HF', 'STO-3G')
	assert d._orders == [0, 1, 2]
	d.prepare(False)

	def _get_Zs_from_file(fn):
		with open(fn) as fh:
			lines = fh.readlines()[-2:]
		zs = [float(_.split()[-1]) for _ in lines]
		return zs

	delta = 0.05
	assert _get_Zs_from_file('multiqm-run/order-0/site-all-cc/run.inp') == [2., 3.]

	assert _get_Zs_from_file('multiqm-run/order-1/site-0-up/run.inp') == [2.+delta, 3.]
	assert _get_Zs_from_file('multiqm-run/order-1/site-0-dn/run.inp') == [2.-delta, 3.]
	assert _get_Zs_from_file('multiqm-run/order-1/site-1-up/run.inp') == [2., 3.+delta]
	assert _get_Zs_from_file('multiqm-run/order-1/site-1-dn/run.inp') == [2., 3.-delta]

	assert _get_Zs_from_file('multiqm-run/order-2/site-0-0-up/run.inp') == [2.+delta, 3.]
	assert _get_Zs_from_file('multiqm-run/order-2/site-0-0-dn/run.inp') == [2.-delta, 3.]
	assert _get_Zs_from_file('multiqm-run/order-2/site-1-1-up/run.inp') == [2., 3.+delta]
	assert _get_Zs_from_file('multiqm-run/order-2/site-1-1-dn/run.inp') == [2., 3.-delta]
	assert _get_Zs_from_file('multiqm-run/order-2/site-0-1-up/run.inp') == [2.+delta, 3.+delta]
	assert _get_Zs_from_file('multiqm-run/order-2/site-0-1-dn/run.inp') == [2.-delta, 3.-delta]

	assert set(map(os.path.basename, glob.glob('multiqm-run/*'))) == set('order-0 order-1 order-2'.split())
	assert set(map(os.path.basename, glob.glob('multiqm-run/order-0/*'))) == set('site-all-cc'.split())
	assert set(map(os.path.basename, glob.glob('multiqm-run/order-1/*'))) == set('site-0-up site-0-dn site-1-up site-1-dn'.split())
	assert set(map(os.path.basename, glob.glob('multiqm-run/order-2/*'))) == set('site-0-0-up site-0-0-dn site-1-1-up site-1-1-dn site-0-1-up site-0-1-dn'.split())
	shutil.rmtree('multiqm-run')

def test_too_high_order():
	c = mqmc.GaussianCalculator()
	with pytest.raises(NotImplementedError):
		d = mqmd.DerivativeFolders(c, 3, [2, 3], np.array([[0, 0, 1], [0, 0, 2]]), 'HF', 'STO-3G')

def test_element_conversion():
	assert mqmd.Derivatives._Z_to_label(0) == "-"
	assert mqmd.Derivatives._Z_to_label(1) == "H"
	assert mqmd.Derivatives._Z_to_label(2) == "He"
