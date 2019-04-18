#!/usr/bin/env python
import pytest

import numpy as np

import mqm.Derivatives as mqmd
import mqm.Calculator as mqmc

def test_grid():
	c = mqmc.MockCalculator()
	d = mqmd.Derivatives(c, 0, [1, 1], [[0, 0, 1], [0, 0, 2]], 'HF', 'STO-3G')
	coords, weights = d._get_grid()
	center = np.average(coords, axis=0)
	assert center[0] - 0 < 1e-8
	assert center[1] - 0 < 1e-8
	assert center[2] - 1.5 < 1e-8

def test_targets():
	c = mqmc.MockCalculator()
	d = mqmd.Derivatives(c, 0, [2, 2], [[0, 0, 1], [0, 0, 2]], 'HF', 'STO-3G')
	targets = set([tuple(_) for _ in d._enumerate_all_targets()])
	expected = set([(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)])
	assert targets == expected
